#if !defined(XMALLOC_H_)
#define XMALLOC_H_
#if defined(__GNUC__)
#define FNATTR_MALLOC __attribute__((malloc))
#else
#define FNATTR_MALLOC
#endif
void * xmalloc (size_t) FNATTR_MALLOC;
void * xcalloc (size_t, size_t) FNATTR_MALLOC;
void * xrealloc (void*, size_t) FNATTR_MALLOC;
inline void * cudaXmalloc(size_t size);

inline void * cudaXmalloc(size_t size) {
	void * devPtr;
	if(cudaMalloc(&devPtr,size) == cudaErrorMemoryAllocation) {
		perror("Failed cudaXmalloc: ");
		abort();
	}
	return devPtr;
} 

#endif /* XMALLOC_H_ */
