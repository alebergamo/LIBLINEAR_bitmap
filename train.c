#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
#include <limits.h>
#include "linear.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

#define BUFFER_LENGTH 4096

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s type : set type of solver (default 1)\n"
//	"	0 -- L2-regularized logistic regression (primal)\n"
	"	1 -- L2-regularized L2-loss support vector classification (dual)\n"	
//	"	2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	4 -- multi-class support vector classification by Crammer and Singer\n"
//	"	5 -- L1-regularized L2-loss support vector classification\n"
//	"	6 -- L1-regularized logistic regression\n"
	"	7 -- L2-regularized logistic regression (dual)\n"
	"-c cost : set the parameter C (default 1)\n"
	"-d cost : set the parameter C2 (default 1) Usable only with '-m 3'\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n" 
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n" 
	"		where f is the primal function and pos/neg are # of\n" 
	"		positive/negative data (default 0.01)\n"
	"	-s 1, 3, 4 and 7\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
	"		where f is the primal function (default 0.01)\n"
	"-B bias : if bias = 1, instance x becomes [x; 1]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
//	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	"-W weight_file: set weight file\n"
	"-m mode : 1-vs-all mode (default 0) \n"
	"    0 default LIBLINEAR behaviour\n"
	"    1 Cpos=C/num_pos, Cneg=C/num_neg      (and every slack is also multiplied for W[i])\n"
	"    2 Cpos=(C*num_neg)/num_pos, Cneg=C    (and every slack is also multiplied for W[i])\n"
	"    3 Cpos=C, Cneg=C2    (and every slack is also multiplied for W[i])\n"
	"-t num_threads   (usable only in the 1-vs-all mode)\n"
	"-n num_neg_per_class (default 0, i.e. all the examples. Usable only in the 1vsAll mode)\n"
	"-r mode_neg_set_selection\n"
	"-l <list of index classes separated by colums e.g 0,1,2,3,7>\n"
	"   This computes only the specified 1vsAll and save a file for each model <model_file>_cl%%d\n"
	"   Note: you don't have to specify the labels but the index e.g. idx in 0:length(unique(tr_label))\n"
	"-p <fileName> Apply pairwise products to the data. We specify the products in a file (It is an ASCII file containing a matrix [D x 2] of int)\n"
	"-b <b>:<k> Apply b-bit Minwise hashing to the data. N.B.: mod(b*k, 8)==0 and b<=32\n"
	"-f <command> Command to execute right before the data of the loading\n"
	"-g <command> Command to execute right after the data of the loading\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char *line = NULL;
static int max_line_len;


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();

//struct feature_node *x_space;
unsigned char* x_space_bin;
struct parameter param;
struct problem prob;
struct model* model_;
char *weight_file;
int flag_cross_validation;
int nr_fold;
double bias;

int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name);

	printf("Loading the data.. "); fflush(stdout);
	read_problem(input_file_name);
	printf("done.\n"); fflush(stdout);

	error_msg = check_parameter(&prob,&param);

	if(error_msg)
	{
		fprintf(stderr,"Error: %s\n",error_msg);
		exit(1);
	}

	if(flag_cross_validation)
	{
		//do_cross_validation();
		fprintf(stderr,"Error: %s\n", "Cross validation not supported");
		exit(1);
	}
	else
	{
		model_ = train(&prob, &param);
		if (model_) {
			if(save_model(model_file_name, model_))
			{
				fprintf(stderr,"can't save model to file %s\n",model_file_name);
				exit(1);
			}
			free_and_destroy_model(&model_);
		}
	}
	destroy_param(&param);
	free(prob.y);
	free(prob.x_bin);
	free(prob.W);
	free(x_space_bin);
	free(line);

	return 0;
}

//void do_cross_validation()
//{
//	int i;
//	int total_correct = 0;
//	int *target = Malloc(int, prob.l);
//
//	cross_validation(&prob,&param,nr_fold,target);
//
//	for(i=0;i<prob.l;i++)
//		if(target[i] == prob.y[i])
//			++total_correct;
//	printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
//
//	free(target);
//}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout
	char* c = NULL;
	char* c2 = NULL;
	int idx;
	char buffer[4096];

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.C2 = 1;
	param.eps = INF; // see setting below
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	flag_cross_validation = 0;
	weight_file = NULL;
	bias = -1;
	param.mode_1vsAll = 0;
	param.nr_threads = 1;
	param.nr_neg_ex_class = 0;
	param.mode_neg_set_selection = 0;
	param.nr_classes_tolearn = 0; // if 0 we learn all the classes. The value will be changed by linear.c:train
	param.classes_tolearn = NULL;
	param.save_separate_model_files = 0;
	param.model_file_name = NULL;
	param.mode_features = MODE_LINEAR;
	param.mode_pairwise_comb_file_name = NULL;
	param.bBitMinwiseHashing_b = 0;
	param.bBitMinwiseHashing_k = 0;
	param.cmd_before_loading = NULL;
	param.cmd_after_loading = NULL;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'c':
				param.C = atof(argv[i]);
				break;

			case 'd':
				param.C2 = atof(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'B':
				bias = atof(argv[i]);
				break;

			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;

			case 'v':
				flag_cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;

			case 'q':
				print_func = &print_null;
				i--;
				break;

			case 'W':
				weight_file = argv[i];
				break;

			case 'm':
				param.mode_1vsAll = atoi(argv[i]);
				if((param.mode_1vsAll < 0) || (param.mode_1vsAll > 3))
				{
					fprintf(stderr,"1-vs-all mode must be 0, 1, 2, 3\n");
					exit_with_help();
				}
				break;
			case 't':
				param.nr_threads = atoi(argv[i]);
				if(param.nr_threads < 1)
				{
					fprintf(stderr,"num_threads must be >= 1\n");
					exit_with_help();
				}
				break;
			case 'n':
				param.nr_neg_ex_class = atoi(argv[i]);
				break;
			case 'r':
				param.mode_neg_set_selection = atoi(argv[i]);
				break;
			case 'l':
				param.save_separate_model_files = 1;
				/* we assume the following syntax: -l 1,12,32,47,59 */
				c = argv[i];
				while (c != NULL) {
					++param.nr_classes_tolearn;
					c = strchr(c, ',');
					c = (c==NULL) ? NULL : c+1;
				}
				param.classes_tolearn = (int*) malloc(param.nr_classes_tolearn*sizeof(int));
				strcpy(buffer, argv[i]);
				c = buffer;
				idx = 0;
				while (c != NULL) {
					c2 = strchr(c, ',');
					if (c2 != NULL) {
						*c2 = '\0';
					}
					param.classes_tolearn[idx] = atoi(c);
					++idx;
					c = (c2==NULL) ? NULL : c2+1;
				}
				break;
			case 'p':
				param.mode_features = MODE_PAIRWISEPRODUCTS;
				param.mode_pairwise_comb_file_name = argv[i];
				break;
			case 'b':
				strcpy(buffer, argv[i]);
				c = strchr(buffer, ':');
				if (!c) {
					fprintf(stderr,"Syntax: -b <b>:<k>\n");
				}
				*c = '\0';
				++c;
				param.bBitMinwiseHashing_b = atoi(buffer);
				if (param.bBitMinwiseHashing_b > 8*sizeof(unsigned int)) {
					fprintf(stderr,"Parameter b for the b-Bit Minwise Hashing must be <=%d\n", 8*sizeof(unsigned int));
					exit_with_help();
				}
				param.bBitMinwiseHashing_k = atoi(c);
				param.mode_features = MODE_BBITMINWISEHASHING;
				if ((prob.bBitMinwiseHashing_b*prob.bBitMinwiseHashing_k) % 8) {
					fprintf(stderr,"Parameter b and k for the b-Bit Minwise Hashing must be that mod(b*k, 8)==0\n", 8*sizeof(unsigned int));
					exit_with_help();
				}
				break;
			case 'f':
				param.cmd_before_loading = argv[i];
				break;
			case 'g':
				param.cmd_after_loading = argv[i];
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	set_print_string_function(print_func);

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}
	param.model_file_name = (char*) malloc((strlen(model_file_name)+1)*sizeof(char));
	strcpy(param.model_file_name, model_file_name);

	if(param.eps == INF)
	{
		if(param.solver_type == L2R_LR || param.solver_type == L2R_L2LOSS_SVC)
			param.eps = 0.01;
		else if(param.solver_type == L2R_L2LOSS_SVC_DUAL || param.solver_type == L2R_L1LOSS_SVC_DUAL || param.solver_type == MCSVM_CS || param.solver_type == L2R_LR_DUAL)
			param.eps = 0.1;
		else if(param.solver_type == L1R_L2LOSS_SVC || param.solver_type == L1R_LR)
			param.eps = 0.01;
	}
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	FILE *fp = NULL;
	char buffer[BUFFER_LENGTH];
	size_t res;
	unsigned int temp_uint;
	unsigned int n_rows;
	unsigned int n_cols;
	size_t n_bytes_data;
	int* labels;
	int* temp_i;
	int temp_int2, temp_int3, temp_int4;
	float temp_float, temp_float2, temp_float3;
	unsigned int i;

	/* execute cmd_before_loading (if requested) */
	if (param.cmd_before_loading) {
		temp_int2 = system(param.cmd_before_loading);
		printf("\nThe command '%s' has returned the code %d.\n", param.cmd_before_loading,temp_int2);
	}

	/* load the binary data */
	fp = fopen(filename,"rb");
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}
	// read n_rows
	res = fread(&temp_uint, sizeof(unsigned int), 1, fp);
	assert(res > 0);
	n_rows = temp_uint;
	if ((n_rows % 8) != 0) {
		fprintf(stderr, "%s %d %s. The number of dimensions (%lu) must be a multiple of 8", __FILE__, __LINE__, filename, n_rows);
		exit(1);
	}
	// read n_cols
	res = fread(&temp_uint, sizeof(unsigned int), 1, fp);
	assert(res > 0);
	n_cols = temp_uint;
	// read labels
	labels = (int*) malloc(sizeof(int)*n_cols);
	res = fread(labels, sizeof(int), n_cols, fp);
	if(res != n_cols) {
		fprintf(stderr, "%s %d %s. The file does not contains the excepted number of labels (expected=%lu, obtained=%lu)\n", __FILE__, __LINE__, buffer,  n_cols, (res/sizeof(int)));
		exit(1);
	}
	/* Read data. If we downsample the negative set, we keep the same negative set for
	 * all the 1vsAll and we need to learn only a subset of the K classes, we optimize the I/O
	 * loading only the data strictly necessary, miminizing the memory needed for the sw.
	 * If data with the same label is stored in a sequential fashion
	 * (the data is divided into chunks each of which has a label) we also speed-up the loading. */
	if ((param.nr_neg_ex_class > 0) && (param.mode_neg_set_selection > 0) && (param.nr_classes_tolearn > 0)) {
		// **** optimized loading by chunks
		// min/max label
		int max_label = INT_MIN;
		int min_label = INT_MAX;
		for (i=0; i < n_cols; ++i) {
			if (labels[i] > max_label) {
				max_label = labels[i];
			}
			if (labels[i] < min_label) {
				min_label = labels[i];
			}
		}
		// create the mapping label_to_idx
		int max_nr_class = (max_label - min_label + 1);
		int label_to_idx[max_nr_class];
		int idx = 0;
		for (i=0; i < max_nr_class; ++i) {
			label_to_idx[i] = INT_MAX;
		}
		for (i=0; i < n_cols; ++i) {
			if ( label_to_idx[labels[i]-min_label] == INT_MAX) {
				label_to_idx[labels[i]-min_label] = idx;
				++idx;
			}
		}
		int nr_class = idx;
		// structure to load the data
		int nr_example_class_to_load[nr_class]; // if -1 we load everything. we initialize the positive classes to be -1 and the negative to be '-n'
		int nr_example_class_loaded[nr_class]; // we initialize this vector to zero
		int example_to_load[n_cols]; // 0 do not load, 1 load. we do not need to initialize the vector
		for (i=0; i < nr_class; ++i) {
			nr_example_class_to_load[i] = param.nr_neg_ex_class;
			nr_example_class_loaded[i] = 0;
		}
		for (i=0; i < param.nr_classes_tolearn; ++i) {
			nr_example_class_to_load[param.classes_tolearn[i]] = -1;
		}
		// fill out the structures below
		int n_example_to_load = 0;
		for (i=0; i < n_cols; ++i) {
			idx = label_to_idx[labels[i]-min_label];
			if (nr_example_class_to_load[idx] == -1) {
				example_to_load[i] = 1;
				++nr_example_class_loaded[idx];
				++n_example_to_load;
			} else if (nr_example_class_loaded[idx] < nr_example_class_to_load[idx]) {
				example_to_load[i] = 1;
				++nr_example_class_loaded[idx];
				++n_example_to_load;
			} else {
				example_to_load[i] = 0; // no loading
			}
		}
		// double-check
		for (i=0; i < nr_class; ++i) {
			if ((nr_example_class_to_load[i] != -1) && (nr_example_class_loaded[i] < nr_example_class_to_load[i])) {
				fprintf(stderr, "It has been requested to load %d neg_ex/cl but the datafile contains only %d examples for a class\n", nr_example_class_to_load[idx], nr_example_class_loaded[idx]);
				exit(1);
			}
		}
		// allocate the memory
		n_bytes_data = (n_rows/8) * (size_t)n_example_to_load;
		x_space_bin = (unsigned char*) malloc(sizeof(unsigned char)*n_bytes_data);
		// given example_to_load, i.e. the index of the data to load, we finally load the data from the disk
		int idx_start = 0; // Note: regardless what the first example contains, we load it for sure
		int idx_end_old = 0;
		int idx_end = -1;
		idx = 0;
		size_t n_bytes_chunk = 0;
		size_t offset = 0;
		while (idx_end < (int)(n_cols-1)) {
			while (example_to_load[idx] == 0) {
				++idx;
				if (idx >= n_cols) {
					break;
				}
			}
			idx_start = idx;
			// if (idx_start >= n_cols) we can exit
			if (idx >= n_cols) {
				break;
			}
			// ..otherwise we continue
			++idx;
			while (example_to_load[idx] == 1) {
				++idx;
				if (idx >= n_cols) {
					break;
				}
			}
			idx_end_old = idx_end;
			idx_end = idx - 1;
			// move the head and load this chunk from the disk
			n_bytes_chunk = (n_rows/8)*(size_t)(idx_end - idx_start + 1);
			res = fseek(fp, (n_rows/8)*(size_t)(idx_start-idx_end_old-1), SEEK_CUR);
			assert(!res);
			res = fread(x_space_bin+offset, sizeof(unsigned char), n_bytes_chunk, fp);
			printf("%.2f MB ", (float)((double)res/1024/1024)); fflush(stdout);
			if(res != n_bytes_chunk) {
				fprintf(stderr, "Error during loading data (expected=%lu, obtained=%lu)\n", n_bytes_chunk, res);
				exit(1);
			}
			// offset
			offset += n_bytes_chunk;
		}
		// modify labels
		temp_i = (int*) malloc(n_example_to_load*sizeof(int));
		idx = 0;
		for (i=0; i < n_cols; ++i) {
			if (example_to_load[i] == 1) {
				temp_i[idx] = labels[i];
				++idx;
			}
		}
		free(labels);
		labels = temp_i;
		temp_i = NULL;
		// modify n_cols
		n_cols = n_example_to_load;
	} else {
		// **** normal loading
		n_bytes_data = (n_rows/8) * (size_t)n_cols;
		x_space_bin = (unsigned char*) malloc(sizeof(unsigned char)*n_bytes_data);
		res = fread(x_space_bin, sizeof(unsigned char), n_bytes_data, fp);
		printf("Read %lu bytes of data.\n", res);
		if(res != n_bytes_data) {
			fprintf(stderr, "%s %d %s. The file does not contains the excepted number of bytes (expected=%lu, obtained=%lu)\n", __FILE__, __LINE__, buffer, n_bytes_data, res);
			exit(1);
		}
	}
	fclose(fp);

	prob.l = n_cols;
	prob.n = (bias >= 0) ? n_rows+1 : n_rows;
	prob.n_bin = n_rows;
	prob.bias=bias;
	prob.y = Malloc(int,prob.l);
	for (i=0; i < prob.l; ++i) {
		prob.y[i] = (int) labels[i];
	}
	prob.x_bin = (unsigned char**) malloc(sizeof(unsigned char*)*n_cols);
	for(i=0; i < n_cols; i++) {
		prob.x_bin[i] = &x_space_bin[(size_t)i * (n_rows/8)];
	}
	prob.W = Malloc(double,prob.l);
	prob.pairwise_comb = NULL;
	prob.mode_features = param.mode_features;
	prob.bBitMinwiseHashing_b = param.bBitMinwiseHashing_b;
	prob.bBitMinwiseHashing_k = param.bBitMinwiseHashing_k;

	/* load the weights */
	if(weight_file)
	{
		fp = fopen(weight_file,"r");
		if(fp == NULL) {
			fprintf(stderr,"can't open weights file %s\n",weight_file);
			exit(1);
		}
		for(i=0;i<prob.l;i++)
			fscanf(fp,"%lf",&prob.W[i]);
		fclose(fp);
	}
	else
	{
		for(i=0;i<prob.l;i++)
			prob.W[i] = 1;
	}

	/* specific-actions for the mode */
	if (param.mode_features == MODE_PAIRWISEPRODUCTS) {
		fp = fopen(param.mode_pairwise_comb_file_name, "r");
		if(fp == NULL) {
			fprintf(stderr,"can't open pairwise-products file %s\n",param.mode_pairwise_comb_file_name);
			exit(1);
		}
		temp_int3 = 0;
		temp_int4 = 1;
		while (temp_int4 != EOF) {
			temp_int4 = fscanf(fp,"%f", &temp_float);
			++temp_int3;
		}
		fclose(fp);
		--temp_int3;
		prob.n_proj = temp_int3/2;
		prob.n = prob.n_proj + (prob.bias>=0 ? 1 : 0);
		// create the vector
		prob.pairwise_comb = Malloc(int, temp_int3);
		fp = fopen(param.mode_pairwise_comb_file_name, "r");
		for (i=0; i<temp_int3; ++i) {
			fscanf(fp,"%f", &temp_float);
			prob.pairwise_comb[i] = (int) temp_float;
		}
		fclose(fp);
	} else if (param.mode_features == MODE_BBITMINWISEHASHING) {
		// double-check
		if ((prob.bBitMinwiseHashing_b*prob.bBitMinwiseHashing_k) != prob.n_bin) {
			fprintf(stderr, "%s %d %s. The file does not contains the excepted number of features (expected from the bBit-Minwise-Hashing-options=%u, obtained=%u)\n", __FILE__, __LINE__, buffer,  (unsigned int) param.bBitMinwiseHashing_b*param.bBitMinwiseHashing_k, (unsigned int) prob.n_bin);
			exit(1);
		}
		// space dims is K*(2^B)
		prob.n_proj = (int) prob.bBitMinwiseHashing_k*pow(2,prob.bBitMinwiseHashing_b);
		prob.n = prob.n_proj + (prob.bias>=0 ? 1 : 0);
	}


	/* execute cmd_after_loading (if requested) */
	if (param.cmd_after_loading) {
		temp_int2 = system(param.cmd_after_loading);
		printf("The command '%s' has returned the code %d.\n", param.cmd_after_loading,temp_int2);
	}

	/* free */
	free(labels);
}
