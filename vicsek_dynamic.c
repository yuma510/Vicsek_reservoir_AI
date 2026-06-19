#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#define PI (4.0 * atan(1.0))

typedef struct {
    double x, y, theta;
} particle_t;

/* ── xoshiro256** RNG + Box-Muller ─────────────────────────────────────── */

typedef struct {
    uint64_t s[4];
    int      iset;
    double   gset;
} xrng_t;

/* splitmix64: expand a 64-bit seed into 256-bit xoshiro state */
static void xrng_seed(xrng_t *r, uint64_t seed)
{
    r->iset = 0;
    for (int i = 0; i < 4; i++) {
        seed += UINT64_C(0x9e3779b97f4a7c15);
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        r->s[i] = z ^ (z >> 31);
    }
}

static inline uint64_t rotl64(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static uint64_t xrng_next_raw(xrng_t *r)
{
    const uint64_t result = rotl64(r->s[1] * 5, 7) * 9;
    const uint64_t t      = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3]  = rotl64(r->s[3], 45);
    return result;
}

/* uniform [0, 1) */
static double xrng_uniform(xrng_t *r)
{
    return (xrng_next_raw(r) >> 11) * (1.0 / (UINT64_C(1) << 53));
}

/* standard normal via Box-Muller (caches one value) */
static double xrng_normal(xrng_t *r)
{
    if (r->iset) { r->iset = 0; return r->gset; }
    double v1, v2, rsq;
    do {
        v1  = 2.0 * xrng_uniform(r) - 1.0;
        v2  = 2.0 * xrng_uniform(r) - 1.0;
        rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    double fac = sqrt(-2.0 * log(rsq) / rsq);
    r->gset = v1 * fac;
    r->iset = 1;
    return v2 * fac;
}

/* ── Cell-linked-list helpers ───────────────────────────────────────────── */

static int icell1(int ix, int iy, int M)
{
    return 1 + ((ix - 1 + M) % M) + ((iy - 1 + M) % M) * M;
}

static void build_cell_map(int M, int *map)
{
    for (int iy = 1; iy <= M; iy++) {
        for (int ix = 1; ix <= M; ix++) {
            int imap  = (icell1(ix, iy, M) - 1) * 4;
            int ix_r  = (ix == M) ? 1 : ix + 1;
            int ix_l  = (ix == 1) ? M : ix - 1;
            int iy_u  = (iy == M) ? 1 : iy + 1;
            map[imap + 1] = icell1(ix_r, iy,   M);
            map[imap + 2] = icell1(ix_r, iy_u, M);
            map[imap + 3] = icell1(ix,   iy_u, M);
            map[imap + 4] = icell1(ix_l, iy_u, M);
        }
    }
}

static void build_linked_list(particle_t *particle, int N, int M, int ncell,
                               double boxsize, int *list, int *head)
{
    for (int icell = 1; icell <= ncell; icell++)
        head[icell] = -1;

    double celli = (double)M;
    for (int i = 0; i < N; i++) {
        int icell = 1 + (int)((particle[i].x / boxsize) * celli)
                      + (int)((particle[i].y / boxsize) * celli) * M;
        list[i]   = head[icell];
        head[icell] = i;
    }
}

static void compute_interactions(double rcut, particle_t *particle, double *ft,
                                  int *list, int *head, int *map,
                                  int N, int ncell, double boxsize, int *A)
{
    double count[N];
    for (int i = 0; i < N; i++) { ft[i] = 0.0; count[i] = 0.0; }

    for (int icell = 1; icell <= ncell; icell++) {
        int i = head[icell];
        while (i > -1) {
            double fi = ft[i];

            for (int j = list[i]; j > -1; j = list[j]) {
                double dx  = particle[j].x - particle[i].x;
                double dy  = particle[j].y - particle[i].y;
                double rij = sqrt(dx * dx + dy * dy);
                if (rij < rcut) {
                    double fij  = sin(particle[j].theta - particle[i].theta);
                    fi         += fij;
                    ft[j]      -= fij;
                    count[i]   += 1.0;
                    count[j]   += 1.0;
                    A[i * N + j] = 1;
                }
            }

            int jcell0 = 4 * (icell - 1);
            for (int nb = 1; nb <= 4; nb++) {
                int jcell = map[jcell0 + nb];
                for (int j = head[jcell]; j > -1; j = list[j]) {
                    double rxij = particle[j].x - particle[i].x;
                    double ryij = particle[j].y - particle[i].y;
                    if (rxij >  boxsize * 0.5) rxij -= boxsize;
                    if (rxij <= -boxsize * 0.5) rxij += boxsize;
                    if (ryij >  boxsize * 0.5) ryij -= boxsize;
                    if (ryij <= -boxsize * 0.5) ryij += boxsize;
                    double rij = sqrt(rxij * rxij + ryij * ryij);
                    if (rij < rcut) {
                        double fij  = sin(particle[j].theta - particle[i].theta);
                        fi         += fij;
                        ft[j]      -= fij;
                        count[i]   += 1.0;
                        count[j]   += 1.0;
                        A[i * N + j] = 1;
                    }
                }
            }
            ft[i] = fi;
            i = list[i];
        }
    }

    for (int i = 0; i < N; i++)
        if (count[i] > 0.0) ft[i] /= count[i];
}

/* ── I/O helpers ────────────────────────────────────────────────────────── */

static void ensure_directory(const char *path)
{
    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        perror("mkdir");
        exit(1);
    }
}

static void create_output_dir(const char *base_dir, char *dirname, size_t size)
{
    time_t now = time(NULL);
    struct tm *tm_now = localtime(&now);
    char timestamp[32];

    ensure_directory(base_dir);
    if (strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_now) == 0) {
        fprintf(stderr, "strftime failed\n");
        exit(1);
    }
    snprintf(dirname, size, "%s/%s", base_dir, timestamp);
    int attempt = 0;
    while (mkdir(dirname, 0755) != 0) {
        if (errno != EEXIST) { perror("mkdir"); exit(1); }
        attempt++;
        snprintf(dirname, size, "%s/%s_%d", base_dir, timestamp, attempt);
    }
}

static void load_input(const char *filename, double *u, int n)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Error opening %s\n", filename); exit(1); }
    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%lf", &u[i]) != 1) {
            fprintf(stderr, "Error reading input at line %d\n", i + 1);
            fclose(fp);
            exit(1);
        }
    }
    fclose(fp);
}

static void write_params(const char *filename, int model, int N, double boxsize,
                          long ntime, int utime, double h1, double v0, double sgm,
                          double K, double F, double c, double rcut, double rho,
                          long seed_noise, long seed_pos, long seed_nf)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Error opening %s\n", filename); return; }
    fprintf(fp,
        "{\n"
        "  \"model\": %d,\n"
        "  \"N\": %d,\n"
        "  \"boxsize\": %.6f,\n"
        "  \"ntime\": %ld,\n"
        "  \"utime\": %d,\n"
        "  \"h1\": %.6f,\n"
        "  \"v0\": %.6f,\n"
        "  \"sgm\": %.6f,\n"
        "  \"K\": %.6f,\n"
        "  \"F\": %.6f,\n"
        "  \"c\": %.6f,\n"
        "  \"rcut\": %.6f,\n"
        "  \"rho\": %.6f,\n"
        "  \"seed_noise\": %ld,\n"
        "  \"seed_pos\": %ld,\n"
        "  \"seed_nf\": %ld\n"
        "}\n",
        model, N, boxsize, ntime, utime, h1, v0, sgm, K, F, c, rcut, rho,
        seed_noise, seed_pos, seed_nf);
    fclose(fp);
}

static void initialize_particles(particle_t *particle, int N, double boxsize, xrng_t *r)
{
    for (int i = 0; i < N; i++) {
        particle[i].x     = xrng_uniform(r) * boxsize;
        particle[i].y     = xrng_uniform(r) * boxsize;
        particle[i].theta = xrng_uniform(r) * 2.0 * PI - PI;
    }
}

static void apply_periodic_boundary(particle_t *p, double boxsize)
{
    if (p->x <  0.0)     p->x += boxsize;
    if (p->x >= boxsize) p->x -= boxsize;
    if (p->y <  0.0)     p->y += boxsize;
    if (p->y >= boxsize) p->y -= boxsize;
}

static void normalize_theta(double *theta)
{
    if (*theta >  PI) *theta -= 2.0 * PI;
    if (*theta < -PI) *theta += 2.0 * PI;
}

/* ── JSON 設定ファイル ──────────────────────────────────────────────────── */

#define CFG_STR_LEN 512

typedef struct {
    char   input_file[CFG_STR_LEN];
    char   output_base[CFG_STR_LEN];
    int    N, utime;
    long   ntime;
    double boxsize, rho, v0, K, F, c, h1, rcut, sgm;
    long   seed_noise, seed_pos, seed_nf;  /* -1 = 未指定（デフォルト使用） */
} cfg_t;

static cfg_t make_default_cfg(void)
{
    cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    strncpy(cfg.input_file,  "tmp/narma10_input_0.0:0.5_seed666.dat", CFG_STR_LEN - 1);
    strncpy(cfg.output_base, "data", CFG_STR_LEN - 1);
    cfg.N          = 500;
    cfg.utime      = 10;
    cfg.ntime      = 120000;
    cfg.boxsize    = 15.8;
    cfg.rho        = 2.0;
    cfg.v0         = 0.5;
    cfg.K          = 1.0;
    cfg.F          = 14.3;
    cfg.c          = 0.1;
    cfg.h1         = 0.01;
    cfg.rcut       = 1.0;
    cfg.sgm        = 0.0;
    cfg.seed_noise = -1;
    cfg.seed_pos   = -1;
    cfg.seed_nf    = -1;
    return cfg;
}

static void parse_config(const char *path, cfg_t *cfg)
{
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Cannot open config: %s\n", path); exit(1); }
    char line[1024], buf[CFG_STR_LEN];
    while (fgets(line, sizeof(line), fp)) {
        int iv; long lv; double dv;
        if (sscanf(line, " \"input_file\": \"%511[^\"]\"",  buf) == 1)
            strncpy(cfg->input_file,  buf, CFG_STR_LEN - 1);
        if (sscanf(line, " \"output_base\": \"%511[^\"]\"", buf) == 1)
            strncpy(cfg->output_base, buf, CFG_STR_LEN - 1);
        if (sscanf(line, " \"N\": %d",           &iv) == 1) cfg->N          = iv;
        if (sscanf(line, " \"utime\": %d",       &iv) == 1) cfg->utime      = iv;
        if (sscanf(line, " \"ntime\": %ld",      &lv) == 1) cfg->ntime      = lv;
        if (sscanf(line, " \"boxsize\": %lf",    &dv) == 1) cfg->boxsize    = dv;
        if (sscanf(line, " \"rho\": %lf",        &dv) == 1) cfg->rho        = dv;
        if (sscanf(line, " \"v0\": %lf",         &dv) == 1) cfg->v0         = dv;
        if (sscanf(line, " \"K\": %lf",          &dv) == 1) cfg->K          = dv;
        if (sscanf(line, " \"F\": %lf",          &dv) == 1) cfg->F          = dv;
        if (sscanf(line, " \"c\": %lf",          &dv) == 1) cfg->c          = dv;
        if (sscanf(line, " \"h1\": %lf",         &dv) == 1) cfg->h1         = dv;
        if (sscanf(line, " \"rcut\": %lf",       &dv) == 1) cfg->rcut       = dv;
        if (sscanf(line, " \"sgm\": %lf",        &dv) == 1) cfg->sgm        = dv;
        if (sscanf(line, " \"seed_noise\": %ld", &lv) == 1) cfg->seed_noise = lv;
        if (sscanf(line, " \"seed_pos\": %ld",   &lv) == 1) cfg->seed_pos   = lv;
        if (sscanf(line, " \"seed_nf\": %ld",    &lv) == 1) cfg->seed_nf    = lv;
    }
    fclose(fp);
}

/* ── seed_array ─────────────────────────────────────────────────────────── */

static long seed_array[8] = {10, 11, 12, 13, 14, 15, 16, 17};

int main(int argc, char *argv[])
{
    cfg_t cfg = make_default_cfg();
    if (argc > 1) parse_config(argv[1], &cfg);

    /* seed_array への反映 */
    if (cfg.seed_noise >= 0) seed_array[2] = cfg.seed_noise;
    if (cfg.seed_pos   >= 0) seed_array[3] = cfg.seed_pos;
    if (cfg.seed_nf    >= 0) seed_array[6] = cfg.seed_nf;

    const int model  = 1;
    const int M      = 1;
    const int ncell  = M * M;
    const int mapsiz = 4 * ncell;
    const int ntime  = (int)cfg.ntime;
    const int n_input = ntime / cfg.utime;

    /* --- Allocations --- */
    particle_t *particle = malloc((size_t)cfg.N * sizeof(particle_t));
    double     *ft       = malloc((size_t)cfg.N * sizeof(double));
    int        *list     = malloc((size_t)cfg.N * sizeof(int));
    int        *head     = malloc((size_t)(ncell + 1) * sizeof(int));
    int        *map      = malloc((size_t)(mapsiz + 2) * sizeof(int));
    double     *u        = malloc((size_t)n_input * sizeof(double));
    double     *v        = malloc((size_t)n_input * sizeof(double));
    double     *nf       = malloc((size_t)cfg.N * sizeof(double));
    int        *A        = malloc((size_t)cfg.N * (size_t)cfg.N * sizeof(int));

    if (!particle || !ft || !list || !head || !map || !u || !v || !nf || !A) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    build_cell_map(M, map);
    load_input(cfg.input_file, u, n_input);

    time_t t_start = time(NULL);

    printf("rcut=%f sgm=%f\n", cfg.rcut, cfg.sgm);

    char dirname[512], pos_file[768], params_file[768];
    create_output_dir(cfg.output_base, dirname, sizeof(dirname));
    snprintf(pos_file,    sizeof(pos_file),    "%s/position.dat",      dirname);
    snprintf(params_file, sizeof(params_file), "%s/params_model.json", dirname);

    write_params(params_file, model, cfg.N, cfg.boxsize, (long)ntime, cfg.utime,
                 cfg.h1, cfg.v0, cfg.sgm, cfg.K, cfg.F, cfg.c, cfg.rcut, cfg.rho,
                 seed_array[2], seed_array[3], seed_array[6]);

    FILE *fp1 = fopen(pos_file, "w");
    if (!fp1) { fprintf(stderr, "Error opening %s\n", pos_file); exit(1); }

    /* seed_array[3]: 位置, seed_array[6]: 自然振動数, seed_array[2]: ノイズ */
    xrng_t rng_pos, rng_nf, rng_noise;
    xrng_seed(&rng_pos,   (uint64_t)seed_array[3]);
    xrng_seed(&rng_nf,    (uint64_t)seed_array[6]);
    xrng_seed(&rng_noise, (uint64_t)seed_array[2]);

    initialize_particles(particle, cfg.N, cfg.boxsize, &rng_pos);
    for (int i = 0; i < cfg.N; i++) {
        nf[i] = 2.0 * PI * (xrng_normal(&rng_nf) + 1.0);
    }

    int countloop = 0;
    double tm = ntime * cfg.h1;
    for (double times = 0.0; times < tm - cfg.h1 / 2.0; times += cfg.h1) {
        int t_idx = countloop / cfg.utime;

        memset(A, 0, (size_t)cfg.N * (size_t)cfg.N * sizeof(int));
        build_linked_list(particle, cfg.N, M, ncell, cfg.boxsize, list, head);
        compute_interactions(cfg.rcut, particle, ft, list, head, map,
                             cfg.N, ncell, cfg.boxsize, A);

        v[t_idx] = 4.0 * (u[t_idx] - 0.25);

        for (int m = 0; m < cfg.N; m++) {
            double xi = xrng_normal(&rng_noise);
            particle[m].x += cfg.h1 * cfg.v0 * cos(particle[m].theta);
            particle[m].y += cfg.h1 * cfg.v0 * sin(particle[m].theta);
            particle[m].theta += cfg.h1 * nf[m]
                               + cfg.h1 * cfg.K / cfg.N * ft[m]
                               + cfg.h1 * cfg.F * sin(cfg.c * v[t_idx] - particle[m].theta)
                               + sqrt(cfg.h1) * cfg.sgm * xi;
            normalize_theta(&particle[m].theta);
            apply_periodic_boundary(&particle[m], cfg.boxsize);

            if (isnan(particle[m].x) || isnan(particle[m].y) || isnan(particle[m].theta)) {
                fprintf(stderr,
                        "NaN: sgm=%f loop=%d particle=%d "
                        "x=%f y=%f theta=%f xi=%f nf=%f ft=%f v=%f\n",
                        cfg.sgm, countloop, m,
                        particle[m].x, particle[m].y, particle[m].theta,
                        xi, nf[m], ft[m], v[t_idx]);
                fflush(stderr);
            }
            if ((countloop + 1) % cfg.utime == 0) {
                fprintf(fp1, "%f        %f      %f\n",
                        particle[m].x, particle[m].y, particle[m].theta);
            }
        }
        countloop++;
        if ((countloop % cfg.utime == 0) && fflush(fp1) != 0) {
            fprintf(stderr, "fflush failed (disk full?): rcut=%g sgm=%g loop=%d\n",
                    cfg.rcut, cfg.sgm, countloop);
            fclose(fp1);
            exit(1);
        }
    }
    fclose(fp1);

    printf("duration= %ld\n", (long)(time(NULL) - t_start));
    printf("output base: %s\n", cfg.output_base);

    free(particle); free(ft); free(list); free(head); free(map);
    free(u); free(v); free(nf); free(A);
    return 0;
}
