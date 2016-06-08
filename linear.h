#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#ifdef __cplusplus
extern "C" {
#endif


/*
 * Macro to retrieve the target machine and set the system path separator.
 *
 * TODO These macro should be work with GCC and Visual-C. Other compilers???? cygwin??
*/
#if defined(__unix) || defined(__linux) ||defined(__CYGWIN__) || defined(macintosh) || defined(__APPLE__)
#define SYSTEM_PATH_SEPARATOR '/'
#define SYSTEM_TYPE_NIX
#elif defined(_WIN32)
#define SYSTEM_PATH_SEPARATOR '\\'
#define SYSTEM_TYPE_WIN
#endif
#if defined(__unix) || defined(__linux)
#define SYSTEM_TYPE_LINUX
#endif
#if defined(macintosh) || defined(__APPLE__)
#define SYSTEM_TYPE_MAC
#endif

#define GCC_VERSION (__GNUC__ * 10000 \
                     + __GNUC_MINOR__ * 100 \
                     + __GNUC_PATCHLEVEL__)

#define BUFFER_LENGTH 4096

struct problem
{
	int l; /* num of examples */
	int n; /* final dimensionality of the vector (bias included, mapping included, if present) */
	int n_proj; /* final dimensionality of the data (mapping included if present). Note: this is just 'n-(bias>=0?1:0)' the redondancy is just for our comodity*/
	int *y; /* label */
	unsigned char** x_bin;   /* pointers to x_space_bin */
	unsigned int n_bin;     /* dimensionality of the original data that we load from the disk */
	double bias;            /* < 0 if no bias term; >=0 we add the bias */
	double *W;              /* instance weight */

	int mode_features;

	int* pairwise_comb;    /* if !=NULL this is a (2*n_proj)-length vector containing the pairwise-products of the data */
	int bBitMinwiseHashing_b;
	int bBitMinwiseHashing_k;
};

enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL }; /* solver_type */

enum {MODE_LINEAR, MODE_PAIRWISEPRODUCTS, MODE_BBITMINWISEHASHING}; /* MODE_FEATURES*/

struct parameter
{
	int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	double C2;
	int nr_weight;
	int *weight_label;
	double* weight;

	int mode_1vsAll;
	int nr_threads;
	int nr_neg_ex_class; /* in the 1vsAll this specifies the number of negative examples to use per class. If 0, it uses all the examples*/
	int mode_neg_set_selection;
	int nr_classes_tolearn;
	int* classes_tolearn;
	int save_separate_model_files; /* if 0 (default) normal behaviour, otherwise save a single model file for each class */
	char* model_file_name;

	int mode_features; // MODE_LINEAR: we use the original data;  MODE_PAIRWISEPRODUCTS: we use the pairwise file;  MODE_BBITMINWISEHASHING: b-Bit minwise hashing
	char* mode_pairwise_comb_file_name;

	int bBitMinwiseHashing_b;
	int bBitMinwiseHashing_k;

	char* cmd_before_loading;
	char* cmd_after_loading;

	/* NOTE. if you add a new entry, modify train.c:parse_command_line and matlab/train.c:parse_command_line */
};

struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	double *w;
	int *label;		/* label of each class */
	double bias;
};

struct model* train(struct problem *prob, struct parameter *param);
//void cross_validation(const struct problem *prob, const struct parameter *param, int nr_fold, int *target);

//int predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
//int predict(const struct model *model_, const struct feature_node *x);
//int predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);

int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, int* label);

void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);

const char *check_parameter(const struct problem *prob, const struct parameter *param);
int check_probability_model(const struct model *model);
void set_print_string_function(void (*print_func) (const char*));

static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn);

#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */

