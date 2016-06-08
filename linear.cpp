#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <assert.h>
#include "linear.h"
#include "tron.h"
#include <pthread.h>
typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{   
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif



/* *** structure and functions to support bitmap data *** */
/* This is a table [256 x 9]
 * Each row is associated with a byte (a number 0..255).
 * T[i,0] contains the number of indexes
 * T[i,1]..T[i,T[i,0]] are the real indexes, from 0 (most left bit) to 7 (most right)
 * E.g.: the first rows are:
 * --------------------
 * 0:   0 ? ? ? ? ? ? ? ?
 * 1:   1 7 ? ? ? ? ? ? ?    1==0b00000001
 * 2:   1 6 ? ? ? ? ? ? ?    2==0b00000010
 * 3:   2 6 7 ? ? ? ? ? ?    3==0b00000011
 * etc...
 * 255: 8 1 2 3 4 5 6 7 8    255==0b11111111
 * -------------------
 * */
// private
unsigned int HASHTABLE_BYTE_INDEX[256*9];

/* return 0 for false,  >0 for true*/
unsigned char get_bit(unsigned char* v, unsigned int i) {
	v = v + (i/8);
	switch (i%8) {
	case 0: { return (*v & 0x80); }
	case 1: { return (*v & 0x40); }
	case 2: { return (*v & 0x20); }
	case 3: { return (*v & 0x10); }
	case 4: { return (*v & 0x08); }
	case 5: { return (*v & 0x04); }
	case 6: { return (*v & 0x02); }
	case 7: { return (*v & 0x01); }
	default: { assert(0); }
	}
}

/*set the bit to one */
void set_bit(unsigned char* v, unsigned int i) {
	v = v + (i/8);
	switch (i%8) {
	case 0: { *v |= 0x80; break; }
	case 1: { *v |= 0x40; break; }
	case 2: { *v |= 0x20; break; }
	case 3: { *v |= 0x10; break; }
	case 4: { *v |= 0x08; break; }
	case 5: { *v |= 0x04; break; }
	case 6: { *v |= 0x02; break; }
	case 7: { *v |= 0x01; break; }
	default: { assert(0); }
	}
}

/* TODO to implement for different compilers. E.g. Visual C++ offers __byteswap_ulong  */
unsigned int endian_swap_uint32(unsigned int v) {
	unsigned int res = 0;
#if GCC_VERSION >= 40300
	res = __builtin_bswap32(v);
#else
   // generic implementation
   res = (v >> 24) |
		 ((v<<8) && 0x00FF0000) |
		 ((v>>8) && 0x0000FF00) |
		 (v << 24);
#endif
	return res;
}

/* TODO to implement for different compilers. E.g. Visual C++ offers __byteswap_uint64  */
unsigned long int endian_swap_uint64(unsigned long int v) {
	unsigned long int res = 0;
#if GCC_VERSION >= 40300
	res = __builtin_bswap64(v);
#else
    res = (v >> 56) |
          ((v<<40) && 0x00FF000000000000) |
          ((v<<24) && 0x0000FF0000000000) |
          ((v<<8) && 0x000000FF00000000) |
          ((v>>8) && 0x00000000FF000000) |
          ((v>>24) && 0x0000000000FF0000) |
          ((v>>40) && 0x000000000000FF00) |
          (v << 56);
#endif
	return res;
}

/* public
 * Return an uint such that the most 'value' right bit are set to one
 * */
unsigned int get_uint_mask_right(unsigned int value) {
	unsigned int res = 0xFFFFFFFF;
	if (value==0) {
		res = 0;
	} else {
		res >>= 8*sizeof(unsigned int)-value;
	}
	return res;
}

/* public
 * Initialize the hash table. This function must be run just once
 */
void hashtable_byte_index_init()
{
	unsigned int i;
	unsigned int offset;

	for (i=0; i < 256; ++i) {
		offset = 1;
		if (i & 0x80) { HASHTABLE_BYTE_INDEX[i*9+offset] = 0; ++offset; }
		if (i & 0x40) { HASHTABLE_BYTE_INDEX[i*9+offset] = 1; ++offset; }
		if (i & 0x20) { HASHTABLE_BYTE_INDEX[i*9+offset] = 2; ++offset; }
		if (i & 0x10) { HASHTABLE_BYTE_INDEX[i*9+offset] = 3; ++offset; }
		if (i & 0x08) { HASHTABLE_BYTE_INDEX[i*9+offset] = 4; ++offset; }
		if (i & 0x04) { HASHTABLE_BYTE_INDEX[i*9+offset] = 5; ++offset; }
		if (i & 0x02) { HASHTABLE_BYTE_INDEX[i*9+offset] = 6; ++offset; }
		if (i & 0x01) { HASHTABLE_BYTE_INDEX[i*9+offset] = 7; ++offset; }
		HASHTABLE_BYTE_INDEX[i*9] = offset-1;
	}
}

/*
 * Private.
 * Given 8 bits, it returns the number of bits are set to one (exploiting the hash-table)
 * */
unsigned int hashtable_byte_index_sum(unsigned char b)
{
	return HASHTABLE_BYTE_INDEX[b*9];
}

/*
 * Private.
 * Given 8 bits (uchar 'c'), it writes on the array pointed by 'v_index', the index of the bits that are set to ones; if offset ~= 0, we add offset to the index.
 * The index most left bit is 0 and the most right 7. 'n_index' is the number of bits are set to one.
 * Note: v_index must be pre-allocated and it must contains at least 8 entries.
 * */
void hashtable_byte_index_find(unsigned char b, unsigned int *v_index, unsigned int *n_index, unsigned int offset)
{
	unsigned int i;
	*n_index = HASHTABLE_BYTE_INDEX[b*9];
	for (i=0; i < *n_index; ++i) {
		v_index[i] = HASHTABLE_BYTE_INDEX[b*9+1+i]+offset;
	}
	//memcpy(v_index, &HASHTABLE_BYTE_INDEX[b*9+1], *n_index); // special case if offset==0
}

/**
 * Public.
 * Given a set of bits 'b', consisting of 'n_bytes', it returns the number of bit set to one.
 * If bias>=0 we assume an extra-bit set to one.
*/
unsigned int bitmap_sum(unsigned char* b, unsigned int n_bytes, double bias)
{
	unsigned int i;
	unsigned int s = 0;
	for (i=0; i < n_bytes; ++i) {
		s += hashtable_byte_index_sum(b[i]);
	}
	s += (bias >= 0) ? 1 : 0;
	return s;
}

/**
 * Public.
 * Given a set of bits 'b', consisting of 'n_bytes', it writes on the array pointed by 'v_index', the index of the bits that are set to ones
 * (the leftmost bit has index zero).
 * 'n_index' is the number of bits are set to one.
 * Note: v_index must be pre-allocated and it must contains at least 8*n_bytes entries.
 * If bias>=0 we assume that the data has an extra-dimension which value is set to one.
 */
void bitmap_find(unsigned char* b, unsigned int n_bytes, unsigned int *v_index, unsigned int *n_index, double bias)
{
	unsigned int i;
	unsigned int offset;
	unsigned int temp_uint;

	offset = 0;
	for (i=0; i < n_bytes; ++i) {
		hashtable_byte_index_find(b[i], v_index+offset, &temp_uint, i*8);
		offset += temp_uint;
	}
	if (bias >= 0) {
		v_index[offset] = n_bytes*8;
		++offset;
	}
	*n_index = offset;
}

/* ***** TIMERS *******/
clock_t tic()
{
  return clock();
}

double toc(clock_t t)
{
  clock_t t_now;

  t_now = clock();
  return (double)(t_now - t)/(double)CLOCKS_PER_SEC;
}

/* **** DATA SAMPLING  ***** */
/* returns a n-dimensional array, containing values [0..n-1]
 * perm must be preallocated */
void randperm(int n, int* perm)
{
	int i, j, t;

	for(i=0; i<n; i++)
		perm[i] = i;
	for(i=0; i<n; i++) {
		j = rand()%(n-i)+i;
		t = perm[j];
		perm[j] = perm[i];
		perm[i] = t;
	}
}

problem* create_sampled_problem(const problem* prob, const parameter* param, int idx_class, int nr_class, int* start, int* count)
{
	problem* prob_sampled = NULL;
	int i, j, idx;

	assert(param->nr_neg_ex_class > 0);

	int si = start[idx_class];
	int ei = si+count[idx_class];
	int num_pos = ei - si;
	int num_neg_sampled = (nr_class-1)*param->nr_neg_ex_class;

	/* allocate memory for the sampled problem */
	prob_sampled = Malloc(problem, 1);
	prob_sampled->l = num_pos+num_neg_sampled;
	prob_sampled->n = prob->n;
	prob_sampled->n_bin = prob->n_bin;
	prob_sampled->x_bin = Malloc(unsigned char*, prob_sampled->l);
	prob_sampled->y = Malloc(int, prob_sampled->l);
	prob_sampled->W = Malloc(double, prob_sampled->l);

	/* create the sampled problem */
	idx = 0;
	for (i=0; i < nr_class; ++i) {
		int si = start[i];
		int ei = si+count[i];
		int num_ex = ei - si;

		if (i == idx_class) {
			/* positive class. Use all the examples */
			for (j=si; j < ei; ++j) {
				prob_sampled->x_bin[idx] = prob->x_bin[j];
				prob_sampled->y[idx] = 1;
				prob_sampled->W[idx] = prob->W[j];
				++idx;
			}
		} else {
			/* negative class. Sample the data from each class either randomly (-r 0) or deterministicly (-r 1) */
			int temp_v_int[num_ex];
			if (param->mode_neg_set_selection == 0) {
				randperm(num_ex, temp_v_int);
			} else if (param->mode_neg_set_selection == 1) {
				for (j=0; j < num_ex; ++j) {
					temp_v_int[j] = j;
				}
			} else {
				assert(0);
			}

			for (j=0; j < param->nr_neg_ex_class; ++j) {
				prob_sampled->x_bin[idx] = prob->x_bin[si + temp_v_int[j]];
				prob_sampled->y[idx] = -1;
				prob_sampled->W[idx] = prob->W[si + temp_v_int[j]];
				++idx;
			}
		}
	}

	return prob_sampled;
}

/* **** DATA ELABORATION *** (Polynomial mapping, minwise hashing, etc..) */
/**
 * Public.
 * Given an example, it elaborates the example (e.g. mapping, hashing, etc..), it returns the number of bit set to one.
 * Note: If 'prob.bias>=0' we assume that the data has an extra-dimension which value is set to one.
*/
unsigned int sum_example(unsigned char* b, const struct problem* prob)
{
	unsigned int i;
	unsigned int s;

	if (prob->mode_features == MODE_PAIRWISEPRODUCTS) {
		/* ** map the vector using the pairwise products (this is a surrogate of the polynomial kernel mapping) */
		s = 0;
		for (i=0; i < prob->n_proj; ++i) {
			if (get_bit(b,prob->pairwise_comb[2*i]) && get_bit(b,prob->pairwise_comb[2*i+1])) {
				++s;
			}
		}
		if (prob->bias >= 0) {
			++s;
		}
		return s;
	} else if (prob->mode_features == MODE_BBITMINWISEHASHING) {
		/* Use b-Bit Minwise Hashing */
		s = prob->bBitMinwiseHashing_k;
		return s;
	} else {
		/* ** use the original data */
		s = bitmap_sum(b, prob->n_bin/8, prob->bias);
		return s;
	}
}

/**
 * Public.
 * Given an example, it elaborates the example (e.g. mapping, hashing, etc..)
 * resulting in a temporary vector, finds the nz dimensions and it
 * writes on the array pointed by 'v_index', the index of the bits that are set to ones
 * (the leftmost bit has index zero).
 * 'n_index' is the number of bits are set to one.
 * Note: v_index must be pre-allocated and it must contains at least 'model.n' entries.
 * Note: If 'prob.bias>=0' we assume that the data has an extra-dimension which value is set to one.
 */
void find_idx_dims_example(unsigned char* b, const struct problem* prob, unsigned int *v_index, unsigned int *n_index)
{
	unsigned int i;
	unsigned int idx_k, idx_bit_start, idx_byte_start, n_bits_chunk, n_bytes_chunk, offset, idx, n_bytes_ori_ex, temp_ui, mask_b_bit_right, n_bit_k;
	unsigned int* ui_ptr = NULL;

	if (prob->mode_features == MODE_PAIRWISEPRODUCTS) {
		/* ** map the vector using the pairwise products (this is a surrogate of the polynomial kernel mapping) */
		*n_index = 0;
		for (i=0; i < prob->n_proj; ++i) {
			if (get_bit(b,prob->pairwise_comb[2*i]) && get_bit(b,prob->pairwise_comb[2*i+1])) {
				*v_index = i;
				++(*n_index);
				++v_index;
			}
		}
		if (prob->bias >= 0) {
			*v_index = prob->n-1;
			++(*n_index);
		}
	} else if (prob->mode_features == MODE_BBITMINWISEHASHING) {
		/* Use b-Bit Minwise Hashing */
		*n_index = 0;
		n_bits_chunk = 8*sizeof(unsigned int);
		n_bytes_chunk = sizeof(unsigned int);
		n_bytes_ori_ex = prob->n_bin / 8;
		mask_b_bit_right = get_uint_mask_right(prob->bBitMinwiseHashing_b);
		n_bit_k = (unsigned int) pow(2,prob->bBitMinwiseHashing_b);
		for (idx_k=0; idx_k < prob->bBitMinwiseHashing_k; ++idx_k) {
			idx_bit_start = idx_k*prob->bBitMinwiseHashing_b;
			idx_byte_start = idx_bit_start / 8;
			offset = idx_bit_start % 8;
			if (idx_byte_start >= (n_bytes_ori_ex-n_bytes_chunk)) {
				offset = offset + (n_bytes_chunk - n_bytes_ori_ex + idx_byte_start)*8;
				idx_byte_start = n_bytes_ori_ex - n_bytes_chunk;
			}
			ui_ptr = (unsigned int*) &b[idx_byte_start];
			temp_ui = *ui_ptr;
			temp_ui = endian_swap_uint32(temp_ui); /* in this instruction we are assuming little-endian byte ordering */
			idx = temp_ui >> (n_bits_chunk - offset - prob->bBitMinwiseHashing_b);
			idx &= mask_b_bit_right;
			idx = n_bit_k - idx - 1; // reverse order index
			idx += idx_k*n_bit_k; // offset of the k-th chunk
			*v_index = idx;
			++(*n_index);
			++v_index;
		}
	} else {
		/* ** use the original data */
		bitmap_find(b, prob->n_bin/8, v_index, n_index, prob->bias);
	}
}


/* **** THREADS  ***** */
struct thread_data {
	int idx_cl_start; // indexes of parameters->classes_tolearn
	int idx_cl_end;
	model* output_model;
	int w_size;
	int* start;
	int* count;
	problem prob;
	const parameter* param;
	int* label;
	double* weighted_C;
	int nr_class;
};

char* execute_external_command(const char* cmd) {
	  FILE *in;
	  char buff[8192];
	  char* output = (char*) malloc(8192*sizeof(char));
	  int offset = 0;

	  /* popen creates a pipe so we can read the output
	     of the program we are invoking */
	  if (!(in = popen(cmd, "r"))) {
	    assert(0);
	  }

	  /* read the output of netstat, one line at a time */
	  offset = 0;
	  while (fgets(buff, sizeof(buff), in) != NULL ) {
	    //printf("Output: %s", buff); // DEBUG
	    strcpy(output+offset, buff);
	    offset = offset + strlen(buff);
	  }

	  /* close the pipe */
	  pclose(in);

	  return output;
}

/* total number of CPU-cores */
int get_num_cpu_cores() {
	char* output_cmd = NULL;
	int n;
#if defined(SYSTEM_TYPE_LINUX)
	 output_cmd = execute_external_command("cat /proc/cpuinfo | grep processor | wc -l");
	 n = atoi(output_cmd);
	 free(output_cmd);
	 return n;
#elif defined(SYSTEM_TYPE_MAC)
	 output_cmd = execute_external_command("sysctl -n hw.ncpu");
	 n = atoi(output_cmd);
	 free(output_cmd);
	 return n;
#else
	/* unknown system-type */
	return 1;
#endif
}

/* get machine loading */
double get_machine_loading() {
	int num_cpu_cores;
	double kernel_loading;
	char* temp_s = NULL;

#if defined(SYSTEM_TYPE_LINUX)
	num_cpu_cores = get_num_cpu_cores();
	temp_s = execute_external_command("w | head -n 1 | awk '{print $10}'");
	kernel_loading = (double) atof(temp_s);
	free(temp_s);
	return kernel_loading / (double)num_cpu_cores;
#elif defined(SYSTEM_TYPE_MAC)
	num_cpu_cores = get_num_cpu_cores();
	temp_s = execute_external_command("w | head -n 1 | awk '{print $10}'");
	kernel_loading = (double) atof(temp_s);
	free(temp_s);
	return kernel_loading / (double)num_cpu_cores;
#else
	/* unknown system-type. The machine looks always busy */
	return 1.0;
#endif
}


void* train_one_thread(void* _arg)
{
	thread_data* arg = (thread_data*) _arg;
	int i, idx_cl;
	char buffer[BUFFER_LENGTH];

	// DEBUG
	if (arg->param->save_separate_model_files) {
		// P.S.: in this case model->w must be NULL
		assert(!arg->output_model->w);
	}

	double w[arg->w_size];
	for(idx_cl=arg->idx_cl_start; idx_cl <= arg->idx_cl_end; idx_cl++)
	{
		i = arg->param->classes_tolearn[idx_cl];

		int si = arg->start[i];
		int ei = si + arg->count[i];
		int num_pos = ei - si;
		int num_neg = si + arg->prob.l - ei;

		problem* prob2 = NULL;
		if (arg->param->nr_neg_ex_class > 0) {
			prob2 = create_sampled_problem(&arg->prob, arg->param, i, arg->nr_class, arg->start, arg->count);
			num_neg = (arg->nr_class-1)*arg->param->nr_neg_ex_class;
		} else {
			prob2 = Malloc(problem, 1);
			*prob2 = arg->prob;
			prob2->y = Malloc(int, arg->prob.l);

			int k=0;
			for(; k<si; k++)
				prob2->y[k] = -1;
			for(; k<ei; k++)
				prob2->y[k] = +1;
			for(; k<prob2->l; k++)
				prob2->y[k] = -1;
		}

		info("Training class %d (label: %d, num_pos: %d, num_neg: %d)\n", i, arg->label[i], num_pos, num_neg);

		switch (arg->param->mode_1vsAll) {
			case 0:
				train_one(prob2, arg->param, w, arg->weighted_C[i], arg->param->C);
				break;
			case 1:
				train_one(prob2, arg->param, w, arg->param->C/num_pos, arg->param->C/num_neg);
				break;
			case 2:
				train_one(prob2, arg->param, w, (arg->param->C*num_neg)/num_pos, arg->param->C);
				break;
			case 3:
				train_one(prob2, arg->param, w, arg->param->C, arg->param->C2);
				break;
		}

		if (arg->param->save_separate_model_files) {
			/* -l specified, we save the models on the disk */
			struct model model_temp;
			model_temp = *(arg->output_model);
			model_temp.w = Malloc(double, arg->w_size);
			for(int j=0; j < arg->w_size; j++) {
				model_temp.w[j] = w[j];
			}
			sprintf(buffer, "%s_cl%d", arg->param->model_file_name, i);
			save_model(buffer, &model_temp);
			free(model_temp.w);
		} else {
			for(int j=0; j < arg->w_size; j++) {
				arg->output_model->w[j*arg->nr_class+i] = w[j];
			}
		}


		if (arg->param->nr_neg_ex_class > 0) {
			free(prob2->x_bin);
			free(prob2->y);
			free(prob2->W);
			free(prob2);
		} else {
			free(prob2->y);
			free(prob2);
		}
	}

	return NULL;
}


//class l2r_lr_fun : public function
//
//l2r_lr_fun::l2r_lr_fun(const problem *prob, double Cp, double Cn)
//
//l2r_lr_fun::~l2r_lr_fun()
//
//double l2r_lr_fun::fun(double *w)
//
//void l2r_lr_fun::grad(double *w, double *g)
//
//int l2r_lr_fun::get_nr_variable(void)
//
//void l2r_lr_fun::Hv(double *s, double *Hs)
//
//void l2r_lr_fun::Xv(double *v, double *Xv)
//
//void l2r_lr_fun::XTv(double *v, double *XTv)
//
//class l2r_l2_svc_fun : public function
//
//l2r_l2_svc_fun::l2r_l2_svc_fun(const problem *prob, double Cp, double Cn)
//
//l2r_l2_svc_fun::~l2r_l2_svc_fun()
//
//double l2r_l2_svc_fun::fun(double *w)
//
//void l2r_l2_svc_fun::grad(double *w, double *g)
//
//int l2r_l2_svc_fun::get_nr_variable(void)
//
//void l2r_l2_svc_fun::Hv(double *s, double *Hs)
//
//void l2r_l2_svc_fun::Xv(double *v, double *Xv)
//
//void l2r_l2_svc_fun::subXv(double *v, double *Xv)
//
//void l2r_l2_svc_fun::subXTv(double *v, double *XTv)

// A coordinate descent algorithm for 
// multi-class support vector machines by Crammer and Singer
//
//  min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
//    s.t.     \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
// 
//  where e^m_i = 0 if y_i  = m,
//        e^m_i = 1 if y_i != m,
//  C^m_i = C if m  = y_i, 
//  C^m_i = 0 if m != y_i, 
//  and w_m(\alpha) = \sum_i \alpha^m_i x_i 
//
// Given: 
// x, y, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Appendix of LIBLINEAR paper, Fan et al. (2008)

#define GETI(i) (i)
// To support weights for instances, use GETI(i) (i)

class Solver_MCSVM_CS
{
	public:
		Solver_MCSVM_CS(const problem *prob, int nr_class, double *C, double eps=0.1, int max_iter=100000);
		~Solver_MCSVM_CS();
		void Solve(double *w);
	private:
		void solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new);
		bool be_shrunk(int i, int m, int yi, double alpha_i, double minG);
		double *B, *C, *G;
		int w_size, l;
		int nr_class;
		int max_iter;
		double eps;
		const problem *prob;
};

Solver_MCSVM_CS::Solver_MCSVM_CS(const problem *prob, int nr_class, double *weighted_C, double eps, int max_iter)
{
	this->w_size = prob->n;
	this->l = prob->l;
	this->nr_class = nr_class;
	this->eps = eps;
	this->max_iter = max_iter;
	this->prob = prob;
	this->B = new double[nr_class];
	this->G = new double[nr_class];
	this->C = new double[prob->l];
	for(int i = 0; i < prob->l; i++)
		this->C[i] = prob->W[i] * weighted_C[prob->y[i]];
}

Solver_MCSVM_CS::~Solver_MCSVM_CS()
{
	delete[] B;
	delete[] G;
	delete[] C;
}

int compare_double(const void *a, const void *b)
{
	if(*(double *)a > *(double *)b)
		return -1;
	if(*(double *)a < *(double *)b)
		return 1;
	return 0;
}

void Solver_MCSVM_CS::solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new)
{
	int r;
	double *D;

	clone(D, B, active_i);
	if(yi < active_i)
		D[yi] += A_i*C_yi;
	qsort(D, active_i, sizeof(double), compare_double);

	double beta = D[0] - A_i*C_yi;
	for(r=1;r<active_i && beta<r*D[r];r++)
		beta += D[r];

	beta /= r;
	for(r=0;r<active_i;r++)
	{
		if(r == yi)
			alpha_new[r] = min(C_yi, (beta-B[r])/A_i);
		else
			alpha_new[r] = min((double)0, (beta - B[r])/A_i);
	}
	delete[] D;
}

bool Solver_MCSVM_CS::be_shrunk(int i, int m, int yi, double alpha_i, double minG)
{
	double bound = 0;
	if(m == yi)
		bound = C[GETI(i)];
	if(alpha_i == bound && G[m] < minG)
		return true;
	return false;
}

void Solver_MCSVM_CS::Solve(double *w)
{
	int i, m, s;
	int iter = 0;
	double *alpha =  new double[l*nr_class];
	double *alpha_new = new double[nr_class];
	int *index = new int[l];
	double *QD = new double[l];
	int *d_ind = new int[nr_class];
	double *d_val = new double[nr_class];
	int *alpha_index = new int[nr_class*l];
	int *y_index = new int[l];
	int active_size = l;
	int *active_size_i = new int[l];
	double eps_shrink = max(10.0*eps, 1.0); // stopping tolerance for shrinking
	bool start_from_all = true;
	unsigned int v_index[prob->n];
	unsigned int n_index;
	int j;

	// initial
	for(i=0;i<l*nr_class;i++)
		alpha[i] = 0;
	for(i=0;i<w_size*nr_class;i++)
		w[i] = 0; 
	for(i=0;i<l;i++)
	{
		for(m=0;m<nr_class;m++)
			alpha_index[i*nr_class+m] = m;

		QD[i] = (double) sum_example(prob->x_bin[i], prob);

		active_size_i[i] = nr_class;
		y_index[i] = prob->y[i];
		index[i] = i;
	}

	while(iter < max_iter) 
	{
		double stopping = -INF;
		for(i=0;i<active_size;i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}
		for(s=0;s<active_size;s++)
		{
			i = index[s];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*nr_class];
			int *alpha_index_i = &alpha_index[i*nr_class];

			if(Ai > 0)
			{
				for(m=0;m<active_size_i[i];m++)
					G[m] = 1;
				if(y_index[i] < active_size_i[i])
					G[y_index[i]] = 0;

				find_idx_dims_example(prob->x_bin[i], prob, v_index, &n_index);

				for (j=0; j < n_index; ++j) {
					double *w_i = &w[(v_index[j])*nr_class];
					for(m=0;m<active_size_i[i];m++)
						G[m] += w_i[alpha_index_i[m]];
				}

				double minG = INF;
				double maxG = -INF;
				for(m=0;m<active_size_i[i];m++)
				{
					if(alpha_i[alpha_index_i[m]] < 0 && G[m] < minG)
						minG = G[m];
					if(G[m] > maxG)
						maxG = G[m];
				}
				if(y_index[i] < active_size_i[i])
					if(alpha_i[prob->y[i]] < C[GETI(i)] && G[y_index[i]] < minG)
						minG = G[y_index[i]];

				for(m=0;m<active_size_i[i];m++)
				{
					if(be_shrunk(i, m, y_index[i], alpha_i[alpha_index_i[m]], minG))
					{
						active_size_i[i]--;
						while(active_size_i[i]>m)
						{
							if(!be_shrunk(i, active_size_i[i], y_index[i], 
											alpha_i[alpha_index_i[active_size_i[i]]], minG))
							{
								swap(alpha_index_i[m], alpha_index_i[active_size_i[i]]);
								swap(G[m], G[active_size_i[i]]);
								if(y_index[i] == active_size_i[i])
									y_index[i] = m;
								else if(y_index[i] == m) 
									y_index[i] = active_size_i[i];
								break;
							}
							active_size_i[i]--;
						}
					}
				}

				if(active_size_i[i] <= 1)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;	
					continue;
				}

				if(maxG-minG <= 1e-12)
					continue;
				else
					stopping = max(maxG - minG, stopping);

				for(m=0;m<active_size_i[i];m++)
					B[m] = G[m] - Ai*alpha_i[alpha_index_i[m]] ;

				solve_sub_problem(Ai, y_index[i], C[GETI(i)], active_size_i[i], alpha_new);
				int nz_d = 0;
				for(m=0;m<active_size_i[i];m++)
				{
					double d = alpha_new[m] - alpha_i[alpha_index_i[m]];
					alpha_i[alpha_index_i[m]] = alpha_new[m];
					if(fabs(d) >= 1e-12)
					{
						d_ind[nz_d] = alpha_index_i[m];
						d_val[nz_d] = d;
						nz_d++;
					}
				}

				find_idx_dims_example(prob->x_bin[i], prob, v_index, &n_index);
				for (j=0; j < n_index; ++j) {
					double *w_i = &w[(v_index[j])*nr_class];
					for(m=0;m<nz_d;m++)
						w_i[d_ind[m]] += d_val[m];
				}
			}
		}

		iter++;
		if(iter % 10 == 0)
		{
			info(".");
		}

		if(stopping < eps_shrink)
		{
			if(stopping < eps && start_from_all == true)
				break;
			else
			{
				active_size = l;
				for(i=0;i<l;i++)
					active_size_i[i] = nr_class;
				info("*");
				eps_shrink = max(eps_shrink/2, eps);
				start_from_all = true;
			}
		}
		else
			start_from_all = false;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0;i<w_size*nr_class;i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0;i<l*nr_class;i++)
	{
		v += alpha[i];
		if(fabs(alpha[i]) > 0)
			nSV++;
	}
	for(i=0;i<l;i++)
		v -= alpha[i*nr_class+prob->y[i]];
	info("Objective value = %lf\n",v);
	info("nSV = %d\n",nSV);

	delete [] alpha;
	delete [] alpha_new;
	delete [] index;
	delete [] QD;
	delete [] d_ind;
	delete [] d_val;
	delete [] alpha_index;
	delete [] y_index;
	delete [] active_size_i;
}

// A coordinate descent algorithm for 
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= alpha_i <= upper_bound_i,
// 
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix 
//
// In L1-SVM case:
// 		upper_bound_i = Cp if y_i = 1
// 		upper_bound_i = Cn if y_i = -1
// 		D_ii = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		D_ii = 1/(2*Cp)	if y_i = 1
// 		D_ii = 1/(2*Cn)	if y_i = -1
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
// 
// See Algorithm 3 of Hsieh et al., ICML 2008

#undef GETI
#define GETI(i) (i)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svc(
	const problem *prob, double *w, double eps, 
	double Cp, double Cn, int solver_type)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 1000;
	int *index = new int[l];
	double *alpha = new double[l];
	schar *y = new schar[l];
	int active_size = l;
	unsigned int* v_index = NULL;
	unsigned int n_index;
	int j;

	printf("eps=%f\n", eps);

	v_index = (unsigned int*) malloc(prob->n*sizeof(unsigned int));

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double *diag = new double[l];
	double *upper_bound = new double[l];
	double *C_ = new double[l];
	for(i=0; i<l; i++) 
	{
		if(prob->y[i]>0)
			C_[i] = prob->W[i] * Cp;
		else 
			C_[i] = prob->W[i] * Cn;
		diag[i] = 0.5/C_[i];
		upper_bound[i] = INF;
	}
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
		for(i=0; i<l; i++) 
		{
			diag[i] = 0;
			upper_bound[i] = C_[i];
		}
	}

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		alpha[i] = 0;
		if(prob->y[i] > 0)
		{
			y[i] = +1; 
		}
		else
		{
			y[i] = -1;
		}
		QD[i] = diag[GETI(i)];
		QD[i] += (double) sum_example(prob->x_bin[i], prob);

		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			i = index[s];
			G = 0;
			schar yi = y[i];

			find_idx_dims_example(prob->x_bin[i], prob, v_index, &n_index);
			for (j=0; j < n_index; ++j) {
				G += w[v_index[j]];
			}

			G = G*yi-1;

			C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;

				find_idx_dims_example(prob->x_bin[i], prob, v_index, &n_index);
				for (j=0; j < n_index; ++j) {
					w[v_index[j]] += d;
				}
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	info("nSV = %d\n",nSV);

	delete [] upper_bound;
	delete [] diag;
	delete [] C_;
	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
	free(v_index);
}

// A coordinate descent algorithm for 
// the dual of L2-regularized logistic regression problems
//
//  min_\alpha  0.5(\alpha^T Q \alpha) + \sum \alpha_i log (\alpha_i) + (upper_bound_i - alpha_i) log (upper_bound_i - alpha_i) ,
//    s.t.      0 <= alpha_i <= upper_bound_i,
// 
//  where Qij = yi yj xi^T xj and 
//  upper_bound_i = Cp if y_i = 1
//  upper_bound_i = Cn if y_i = -1
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 5 of Yu et al., MLJ 2010

#undef GETI
#define GETI(i) (i)
// To support weights for instances, use GETI(i) (i)

void solve_l2r_lr_dual(const problem *prob, double *w, double eps, double Cp, double Cn)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double *xTx = new double[l];
	int max_iter = 1000;
	int *index = new int[l];		
	double *alpha = new double[2*l]; // store alpha and C - alpha
	schar *y = new schar[l];	
	int max_inner_iter = 100; // for inner Newton
	double innereps = 1e-2; 
	double innereps_min = min(1e-8, eps);
	double *upper_bound = new double [l];
	unsigned int v_index[prob->n];
	unsigned int n_index;
	int j;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			upper_bound[i] = prob->W[i] * Cp;
			y[i] = +1; 
		}
		else
		{
			upper_bound[i] = prob->W[i] * Cn;
			y[i] = -1;
		}
		alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
		alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];

		xTx[i] = 0;

		find_idx_dims_example(prob->x_bin[i], prob, v_index, &n_index);
		for (j=0; j < n_index; ++j) {
			xTx[i] += 1.0;
			w[v_index[j]] += y[i]*alpha[2*i];
		}

		index[i] = i;
	}

	while (iter < max_iter)
	{
		for (i=0; i<l; i++)
		{
			int j = i+rand()%(l-i);
			swap(index[i], index[j]);
		}
		int newton_iter = 0;
		double Gmax = 0;
		for (s=0; s<l; s++)
		{
			i = index[s];
			schar yi = y[i];
			double C = upper_bound[GETI(i)];
			double ywTx = 0, xisq = xTx[i];

			find_idx_dims_example(prob->x_bin[i], prob, v_index, &n_index);
			for (j=0; j < n_index; ++j) {
				ywTx += w[v_index[j]];
			}

			ywTx *= y[i];
			double a = xisq, b = ywTx;

			// Decide to minimize g_1(z) or g_2(z)
			int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
			if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0) 
			{
				ind1 = 2*i+1;
				ind2 = 2*i;
				sign = -1;
			}

			//  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
			double alpha_old = alpha[ind1];
			double z = alpha_old;
			if(C - z < 0.5 * C) 
				z = 0.1*z;
			double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
			Gmax = max(Gmax, fabs(gp));

			// Newton method on the sub-problem
			const double eta = 0.1; // xi in the paper
			int inner_iter = 0;
			while (inner_iter <= max_inner_iter) 
			{
				if(fabs(gp) < innereps)
					break;
				double gpp = a + C/(C-z)/z;
				double tmpz = z - gp/gpp;
				if(tmpz <= 0) 
					z *= eta;
				else // tmpz in (0, C)
					z = tmpz;
				gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
				newton_iter++;
				inner_iter++;
			}

			if(inner_iter > 0) // update w
			{
				alpha[ind1] = z;
				alpha[ind2] = C-z;

				find_idx_dims_example(prob->x_bin[i], prob, v_index, &n_index);
				for (j=0; j < n_index; ++j) {
					w[v_index[j]] += sign*(z-alpha_old)*yi;
				}
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gmax < eps) 
			break;

		if(newton_iter <= l/10) 
			innereps = max(innereps_min, 0.1*innereps);

	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 0 may be faster (also see FAQ)\n\n");

	// calculate objective value
	
	double v = 0;
	for(i=0; i<w_size; i++)
		v += w[i] * w[i];
	v *= 0.5;
	for(i=0; i<l; i++)
		v += alpha[2*i] * log(alpha[2*i]) + alpha[2*i+1] * log(alpha[2*i+1]) 
			- upper_bound[GETI(i)] * log(upper_bound[GETI(i)]);
	info("Objective value = %lf\n", v);

	delete [] upper_bound;
	delete [] xTx;
	delete [] alpha;
	delete [] y;
	delete [] index;
}

// A coordinate descent algorithm for 
// L1-regularized L2-loss support vector classification
#undef GETI
#define GETI(i) (i)
// To support weights for instances, use GETI(i) (i)

//static void solve_l1r_l2_svc(

#undef GETI
#define GETI(i) (i)
// To support weights for instances, use GETI(i) (i)

//static void solve_l1r_lr(

//// transpose matrix X from row format to column format
//static void transpose(const problem *prob, feature_node **x_space_ret, problem *prob_col)

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn)
{
	double eps=param->eps;
	int pos = 0;
	int neg = 0;
	for(int i=0;i<prob->l;i++)
		if(prob->y[i]==+1)
			pos++;
	neg = prob->l - pos;

	function *fun_obj=NULL;
	switch(param->solver_type)
	{
		case L2R_LR:
		{
			fprintf(stderr,"Error: %s\n","Solver L2R_LR not supported");
			exit(1);
//			fun_obj=new l2r_lr_fun(prob, Cp, Cn);
//			TRON tron_obj(fun_obj, eps*min(pos,neg)/prob->l);
//			tron_obj.set_print_string(liblinear_print_string);
//			tron_obj.tron(w);
//			delete fun_obj;
			break;
		}
		case L2R_L2LOSS_SVC:
		{
			fprintf(stderr,"Error: %s\n","Solver L2R_L2LOSS_SVC not supported");
			exit(1);
//			fun_obj=new l2r_l2_svc_fun(prob, Cp, Cn);
//			TRON tron_obj(fun_obj, eps*min(pos,neg)/prob->l);
//			tron_obj.set_print_string(liblinear_print_string);
//			tron_obj.tron(w);
//			delete fun_obj;
			break;
		}
		case L2R_L2LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L2LOSS_SVC_DUAL);
			break;
		case L2R_L1LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
			break;
		case L1R_L2LOSS_SVC:
		{
			fprintf(stderr,"Error: %s\n","Solver L1R_L2LOSS_SVC not supported");
			exit(1);
//			problem prob_col;
//			feature_node *x_space = NULL;
//			transpose(prob, &x_space ,&prob_col);
//			solve_l1r_l2_svc(&prob_col, w, eps*min(pos,neg)/prob->l, Cp, Cn);
//			delete [] prob_col.y;
//			delete [] prob_col.x;
//			delete [] prob_col.W;
//			delete [] x_space;
			break;
		}
		case L1R_LR:
		{
			fprintf(stderr,"Error: %s\n","Solver L1R_LR not supported");
			exit(1);
//			problem prob_col;
//			feature_node *x_space = NULL;
//			transpose(prob, &x_space ,&prob_col);
//			solve_l1r_lr(&prob_col, w, eps*min(pos,neg)/prob->l, Cp, Cn);
//			delete [] prob_col.y;
//			delete [] prob_col.x;
//			delete [] prob_col.W;
//			delete [] x_space;
			break;
		}
		case L2R_LR_DUAL:
			solve_l2r_lr_dual(prob, w, eps, Cp, Cn);
			break;
		default:
			fprintf(stderr, "Error: unknown solver_type\n");
			break;
	}
}

//
// Remove zero weighed data as libsvm and some liblinear solvers require C > 0.
//
static void remove_zero_weight(problem *newprob, const problem *prob) 
{
	int i;
	int l = 0;
	for(i=0;i<prob->l;i++)
		if(prob->W[i] > 0) l++;
	*newprob = *prob;
	newprob->l = l;
	newprob->x_bin = Malloc(unsigned char*,l);
	newprob->y = Malloc(int,l);
	newprob->W = Malloc(double,l);

	int j = 0;
	for(i=0;i<prob->l;i++)
		if(prob->W[i] > 0)
		{
			newprob->x_bin[j] = prob->x_bin[i];
			newprob->y[j] = prob->y[i];
			newprob->W[j] = prob->W[i];
			j++;
		}
}

//
// Interface functions
//
model* train(problem *prob, parameter *param)
{
	problem newprob;
	remove_zero_weight(&newprob, prob);
	prob = &newprob;
	clock_t timer_time_elapsed;
	char buffer[BUFFER_LENGTH];
	int res;

	int i,j, idx_cl;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);

	int nr_class;
	int *label = NULL;
	int *start = NULL;
	int *count = NULL;
	int *perm = Malloc(int,l);

	// **** initialize the lookup table
	hashtable_byte_index_init();

	// group training data of the same class
	group_classes(prob,&nr_class,&label,&start,&count,perm);

	/* if necessary let's modify param->features_tolearn */
	if (!param->classes_tolearn) {
		param->nr_classes_tolearn = nr_class;
		param->classes_tolearn = (int*) malloc(nr_class*sizeof(int));
		for(i=0; i<nr_class; i++) {
			param->classes_tolearn[i] = i;
		}
	}

	/* model */
	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	// safety check in case (param->nr_neg_ex_class > 0)
	if (param->nr_neg_ex_class > 0) {
		for(i=0; i<nr_class; i++) {
			int si = start[i];
			int ei = si+count[i];
			int num_ex = ei - si;
			if (num_ex < param->nr_neg_ex_class) {
				info("ERROR. The class labeled as %d contains %d examples but you have requested %d negative examples per class.\n", label[i], num_ex, param->nr_neg_ex_class);
				free(model_);
				free(label);
				free(start);
				free(count);
				return NULL;
			}
		}
	}

	model_->nr_class=nr_class;
	model_->label = Malloc(int,nr_class);
	for(i=0;i<nr_class;i++)
		model_->label[i] = label[i];

	// calculate weighted C
	double *weighted_C = Malloc(double, nr_class);
	for(i=0;i<nr_class;i++)
		weighted_C[i] = param->C;
	for(i=0;i<param->nr_weight;i++)
	{
		for(j=0;j<nr_class;j++)
			if(param->weight_label[i] == label[j])
				break;
		if(j == nr_class)
			fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
		else
			weighted_C[j] *= param->weight[i];
	}

	// constructing the subproblem
	unsigned char** x_bin = (unsigned char**) malloc(sizeof(unsigned char*)*l);
	double *W = Malloc(double,l);
	for(i=0;i<l;i++)
	{
		x_bin[i] = prob->x_bin[perm[i]];
		W[i] = prob->W[perm[i]];
	}

	int k;
	problem sub_prob;
	sub_prob = *prob;
	sub_prob.l = l;
	sub_prob.n = n;
	sub_prob.n_bin = prob->n_bin;
	sub_prob.x_bin = Malloc(unsigned char*, sub_prob.l);
	sub_prob.y = Malloc(int,sub_prob.l);
	sub_prob.W = Malloc(double,sub_prob.l);


	for(k=0; k<sub_prob.l; k++)
	{
		sub_prob.x_bin[k] = x_bin[k];
		sub_prob.W[k] = W[k];
	}

	// multi-class svm by Crammer and Singer
	if(param->solver_type == MCSVM_CS)
	{
		model_->w=Malloc(double, n*nr_class);
		for(i=0;i<nr_class;i++)
			for(j=start[i];j<start[i]+count[i];j++)
				sub_prob.y[j] = i;
		Solver_MCSVM_CS Solver(&sub_prob, nr_class, weighted_C, param->eps);
		Solver.Solve(model_->w);
	}
	else
	{
		if(nr_class == 2)
		{
			/* 2-class problem */
			model_->w=Malloc(double, w_size);

			int e0 = start[0]+count[0];
			k=0;
			for(; k<e0; k++)
				sub_prob.y[k] = +1;
			for(; k<sub_prob.l; k++)
				sub_prob.y[k] = -1;

			timer_time_elapsed = tic();
			train_one(&sub_prob, param, &model_->w[0], weighted_C[0], weighted_C[1]);
			info("Training time: %f\n", toc(timer_time_elapsed));
		}
		else
		{
			/* multiclass problem */
			int nr_threads = -1;
			nr_threads = param->nr_threads < nr_class ? param->nr_threads : nr_class;
			if (nr_threads == 1) {
				// ** 1-vs-all NO THREADS
				model_->w=Malloc(double, w_size*(param->save_separate_model_files ? 1 : nr_class));
				double *w=Malloc(double, w_size);
				problem* sub_prob2 = NULL;
				for(idx_cl=0; idx_cl<param->nr_classes_tolearn; ++idx_cl)
				{
					i = param->classes_tolearn[idx_cl];

					int si = start[i];
					int ei = si+count[i];
					int num_pos = ei - si;
					int num_neg = si + sub_prob.l - ei ;

					/*
					 * k         | s_i        | e_i        | prob.l
					 * -1 ... -1 |  1  ...  1 | -1  ... -1 |
					 * */
					if (param->nr_neg_ex_class > 0) {
						sub_prob2 = create_sampled_problem(&sub_prob, param, i, nr_class, start, count);
						num_neg = (nr_class-1)*param->nr_neg_ex_class;
					} else {
						k=0;
						for(; k<si; k++)
							sub_prob.y[k] = -1;
						for(; k<ei; k++)
							sub_prob.y[k] = +1;
						for(; k<sub_prob.l; k++)
							sub_prob.y[k] = -1;
						sub_prob2 = &sub_prob;
					}


					info("Training class %d (label: %d, num_pos: %d, num_neg: %d)\n", i, label[i], num_pos, num_neg);
					switch (param->mode_1vsAll) {
						case 0:
							train_one(sub_prob2, param, w, weighted_C[i], param->C);
							break;
						case 1:
							train_one(sub_prob2, param, w, param->C/num_pos, param->C/num_neg);
							break;
						case 2:
							train_one(sub_prob2, param, w, (param->C*num_neg)/num_pos, param->C);
							break;
						case 3:
							train_one(sub_prob2, param, w, param->C, param->C2);
							break;
					}

					if (param->save_separate_model_files) {
						/* -l specified, we save the models on the disk */
						for(int j=0; j<w_size; j++) {
							model_->w[j] = w[j];
						}
						sprintf(buffer, "%s_cl%d", param->model_file_name, i);
						save_model(buffer, model_);
					} else {
						for(int j=0; j<w_size; j++) {
							model_->w[j*nr_class+i] = w[j];
						}
					}

					if (param->nr_neg_ex_class > 0) {
						free(sub_prob2->x_bin);
						free(sub_prob2->y);
						free(sub_prob2->W);
						free(sub_prob2);
					}
				}
				free(w);
			} else {
				// ** 1-vs-all WITH THREADS
				info("PThreads activated with %d threads.\n", nr_threads); fflush(stdout);
				if (param->save_separate_model_files) {
					model_->w = NULL; // N.B.: in this case the thread must create its own model_->w
				} else {
					model_->w = Malloc(double, w_size*nr_class);
				}
				// arguments to the thread in the shared memory
				thread_data arg[nr_threads];
				pthread_t   threads[nr_threads];
				int num_classes_per_thread = (int) ceil(param->nr_classes_tolearn / (double)nr_threads);
				for (int t=0; t < nr_threads; ++t) {
					// arguments to the thread
					arg[t].idx_cl_start = t*num_classes_per_thread;
					arg[t].idx_cl_end = (t+1)*num_classes_per_thread-1;
					if (arg[t].idx_cl_end >= param->nr_classes_tolearn) {
						arg[t].idx_cl_end = param->nr_classes_tolearn-1;
					}
					arg[t].output_model = model_;
					arg[t].w_size = w_size;
					arg[t].start = start;
					arg[t].count = count;
					arg[t].prob = sub_prob;
					arg[t].param = param;
					arg[t].label = label;
					arg[t].weighted_C = weighted_C;
					arg[t].nr_class = nr_class;
					// we start the threads
					res = pthread_create(&threads[t], NULL, train_one_thread, (void*) &arg[t]);
					if (res) {
						printf("ERROR: return code from pthread_create(..) is %d\n", res);
						exit(-1);
					}
				}

				// wait
				for (int t=0; t < nr_threads; ++t) {
					res = pthread_join(threads[t], NULL);
					if (res) {
						printf("ERROR: return code from pthread_join(..) is %d\n", res);
						exit(-1);
					}
				}
			}
		}

	}

	free(x_bin);
	free(W);
	free(label);
	free(start);
	free(count);
	free(perm);
	free(sub_prob.x_bin);
	free(sub_prob.y);
	free(sub_prob.W);
	free(weighted_C);
	free(newprob.x_bin);
	free(newprob.y);
	free(newprob.W);
	free(newprob.pairwise_comb);

	if (param->save_separate_model_files) {
		free_and_destroy_model(&model_);
		return NULL;
	} else {
		return model_;
	}
}

//void cross_validation(const problem *prob, const parameter *param, int nr_fold, int *target)

//int predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)

//int predict(const model *model_, const feature_node *x)

//int predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)

static const char *solver_type_table[]=
{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
	"L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL", NULL
};


/*
 * FORMAT MODEL FILE:
 * <solver type>: int
 * <nr_class>: int
 * <labels>: 'nr_class'-length vector of int
 * <nr_feature>: int
 * <bias>: double
 * <Wb>: 'nr_feature*nr_class' float matrix in row-order
 *
 * FORMAT ORIGINAL VERSION:
 * See original LIBLINEAR. It is just a simple text file...
 * */
int save_model(const char *model_file_name, const struct model *model_)
{
	int nr_feature = model_->nr_feature;
	const parameter& param = model_->param;
	int n, w_size, nr_w;
	size_t res_sizet;
	float* temp_f_ptr = NULL;
	unsigned int num_elements_Wb, i;

	if(model_->bias >= 0)
		n=nr_feature+1;
	else
		n=nr_feature;
	w_size = n;

	if((model_->nr_class==2 && model_->param.solver_type != MCSVM_CS) || (param.save_separate_model_files)) {
		nr_w=1;
	} else {
		nr_w=model_->nr_class;
	}

	/* open the output fileName */
	FILE *fp = fopen(model_file_name,"wb");
	if(fp == NULL) {
		printf("ERROR opening %s for writing.\nExit.\n", model_file_name);
		return -1;
	}
	printf("Writing model to file %s\n", model_file_name);

	/* <solver type>: int */
	res_sizet = fwrite(&param.solver_type, sizeof(int), 1, fp);
	assert(res_sizet == 1);
	/* <nr_class>: int */
	res_sizet = fwrite(&model_->nr_class, sizeof(int), 1, fp);
	assert(res_sizet == 1);
	/* <labels>: 'nr_class'-length vector of int */
	res_sizet = fwrite(model_->label, sizeof(int), model_->nr_class, fp);
	assert(res_sizet == model_->nr_class);
	/* <nr_feature>: int */
	res_sizet = fwrite(&n, sizeof(int), 1, fp);
	assert(res_sizet == 1);
	/* <bias>: double */
	res_sizet = fwrite(&model_->bias, sizeof(double), 1, fp);
	assert(res_sizet == 1);
	/* <Wb>: 'nr_feature*nr_class' float matrix in row-order */
	num_elements_Wb = n*nr_w;
	temp_f_ptr = (float*) malloc(num_elements_Wb*sizeof(float)); // let's convert Wb in float
	for (i=0; i < num_elements_Wb; ++i) {
		temp_f_ptr[i] = (float) model_->w[i];
	}
	res_sizet = fwrite(temp_f_ptr, sizeof(float), num_elements_Wb, fp);
	assert(res_sizet == num_elements_Wb);
	free(temp_f_ptr);

	/* close the file */
	fclose(fp);

	return 0;

	/* ORIGINAL VERSION */
//	int i;
//	int nr_feature=model_->nr_feature;
//	int n;
//	const parameter& param = model_->param;
//
//	if(model_->bias>=0)
//		n=nr_feature+1;
//	else
//		n=nr_feature;
//	int w_size = n;
//	FILE *fp = fopen(model_file_name,"w");
//	if(fp==NULL) return -1;
//
//	int nr_w;
//	if((model_->nr_class==2 && model_->param.solver_type != MCSVM_CS) || (param.save_separate_model_files))
//		nr_w=1;
//	else
//		nr_w=model_->nr_class;
//
//	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
//	fprintf(fp, "nr_class %d\n", model_->nr_class);
//	fprintf(fp, "label");
//	for(i=0; i<model_->nr_class; i++)
//		fprintf(fp, " %d", model_->label[i]);
//	fprintf(fp, "\n");
//
//	fprintf(fp, "nr_feature %d\n", nr_feature);
//
//	fprintf(fp, "bias %.16g\n", model_->bias);
//
//	fprintf(fp, "w\n");
//	for(i=0; i<w_size; i++)
//	{
//		int j;
//		for(j=0; j<nr_w; j++)
//			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
//		fprintf(fp, "\n");
//	}
//
//	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
//	else return 0;
}

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				free(model_->label);
				free(model_);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			fscanf(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free(model_);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2 && param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fscanf(fp, "%lf ", &model_->w[i*nr_w+j]);
		fscanf(fp, "\n");
	}
	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
	if (param->classes_tolearn)
		free(param->classes_tolearn);
	if (param->model_file_name)
		free(param->model_file_name);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->solver_type != L2R_LR
		&& param->solver_type != L2R_L2LOSS_SVC_DUAL
		&& param->solver_type != L2R_L2LOSS_SVC
		&& param->solver_type != L2R_L1LOSS_SVC_DUAL
		&& param->solver_type != MCSVM_CS
		&& param->solver_type != L1R_L2LOSS_SVC
		&& param->solver_type != L1R_LR
		&& param->solver_type != L2R_LR_DUAL)
		return "unknown solver type";

	return NULL;
}

int check_probability_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_LR ||
			model_->param.solver_type==L2R_LR_DUAL ||
			model_->param.solver_type==L1R_LR);
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL) 
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

