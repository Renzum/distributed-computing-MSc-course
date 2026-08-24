struct Arguments {
    int domain_width = 0;
    int domain_height = 0;
    int iterations = 1000;
    bool print_final = false;
};

Arguments get_cmd_args(int argc, char *argv[]);