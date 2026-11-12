#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/syscall.h>

typedef struct {
    uint64_t samples;
    uint64_t hits;
    uint64_t worker_index;
    uint64_t random_state;
} WorkerArguments;

static long current_thread_id(void) {
    const long thread_id = syscall(SYS_gettid);

    if (thread_id == -1) {
        perror("syscall(SYS_gettid)");
    }
    return thread_id;
}

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t value = (*state += UINT64_C(0x9e3779b97f4a7c15));

    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static double random_unit_interval(uint64_t *state) {
    /* 53 случайных бита позволяют представить число double в диапазоне [0, 1). */
    return (double)(splitmix64_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

static void *monte_carlo_worker(void *raw_arguments) {
    WorkerArguments *arguments = raw_arguments;
    uint64_t hits = 0;

    for (uint64_t sample = 0; sample < arguments->samples; sample++) {
        const double x = 2.0 * random_unit_interval(&arguments->random_state) - 1.0;
        const double y = 2.0 * random_unit_interval(&arguments->random_state) - 1.0;

        if (x * x + y * y <= 1.0) {
            hits++;
        }
    }

    arguments->hits = hits;
    return NULL;
}

static int parse_radius(const char *text, double *radius) {
    char *end = NULL;

    errno = 0;
    const double parsed = strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !isfinite(parsed) || parsed <= 0.0) {
        return -1;
    }

    *radius = parsed;
    return 0;
}

static int parse_positive_uint64(const char *text, uint64_t *value) {
    char *end = NULL;

    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed == 0) {
        return -1;
    }

    *value = (uint64_t)parsed;
    return 0;
}

static void print_usage(const char *program_name) {
    fprintf(stderr, "Использование: %s <радиус> <испытания> <максимум_потоков>\n", program_name);
    fprintf(stderr, "  радиус             положительный радиус окружности\n");
    fprintf(stderr, "  испытания          положительное число испытаний Монте-Карло\n");
    fprintf(stderr, "  максимум_потоков   максимальное число одновременно работающих потоков\n");
}

int main(int argc, char **argv) {
    double radius;
    uint64_t rounds;
    uint64_t requested_threads;

    if (argc != 4) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (parse_radius(argv[1], &radius) != 0) {
        fprintf(stderr, "Некорректный радиус: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    if (parse_positive_uint64(argv[2], &rounds) != 0) {
        fprintf(stderr, "Некорректное число испытаний: %s\n", argv[2]);
        return EXIT_FAILURE;
    }
    if (parse_positive_uint64(argv[3], &requested_threads) != 0) {
        fprintf(stderr, "Некорректное максимальное число потоков: %s\n", argv[3]);
        return EXIT_FAILURE;
    }

    if (requested_threads > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "Ограничение числа потоков слишком велико для этой платформы.\n");
        return EXIT_FAILURE;
    }
    if (requested_threads > SIZE_MAX / sizeof(pthread_t) ||
        requested_threads > SIZE_MAX / sizeof(WorkerArguments)) {
        fprintf(stderr, "Запрошенное ограничение числа потоков слишком велико.\n");
        return EXIT_FAILURE;
    }

    const size_t thread_count = (size_t)(requested_threads < rounds ? requested_threads : rounds);
    pthread_t *threads = calloc(thread_count, sizeof(*threads));
    WorkerArguments *arguments = calloc(thread_count, sizeof(*arguments));
    if (threads == NULL || arguments == NULL) {
        perror("calloc");
        free(threads);
        free(arguments);
        return EXIT_FAILURE;
    }

    const uint64_t base_samples = rounds / thread_count;
    const uint64_t extra_samples = rounds % thread_count;
    size_t created_threads = 0;

    for (size_t index = 0; index < thread_count; index++) {
        arguments[index].samples = base_samples + (index < extra_samples ? 1U : 0U);
        arguments[index].worker_index = (uint64_t)index;
        arguments[index].random_state = UINT64_C(0x6a09e667f3bcc909) +
                                        UINT64_C(0x9e3779b97f4a7c15) * (uint64_t)(index + 1);
        if (arguments[index].random_state == 0) {
            arguments[index].random_state = UINT64_C(1);
        }

        const int create_status = pthread_create(&threads[index], NULL, monte_carlo_worker, &arguments[index]);
        if (create_status != 0) {
            fprintf(stderr, "Ошибка создания потока: %s\n", strerror(create_status));
            break;
        }
        created_threads++;
    }

    int failed = created_threads != thread_count;
    for (size_t index = 0; index < created_threads; index++) {
        const int join_status = pthread_join(threads[index], NULL);
        if (join_status != 0) {
            fprintf(stderr, "Ошибка ожидания потока: %s\n", strerror(join_status));
            failed = 1;
        }
    }

    if (failed) {
        free(threads);
        free(arguments);
        return EXIT_FAILURE;
    }

    uint64_t total_hits = 0;
    for (size_t index = 0; index < thread_count; index++) {
        total_hits += arguments[index].hits;
    }

    const long double hit_ratio = (long double)total_hits / (long double)rounds;
    const long double estimated_area = 4.0L * (long double)radius * (long double)radius * hit_ratio;

    printf("процесс=%ld потоков=%zu испытаний=%" PRIu64 " попаданий=%" PRIu64 "\n",
           (long)getpid(), thread_count, rounds, total_hits);
    printf("Оценка площади окружности = %.12Lf\n", estimated_area);

    free(threads);
    free(arguments);
    return EXIT_SUCCESS;
}
