#ifndef LAMBDA_LANCZOS_MACRO_H
#define LAMBDA_LANCZOS_MACRO_H_

#if defined(LAMBDA_LANCZOS_STDPAR_SEQ)
#define LAMBDA_LANCZOS_STDPAR_POLICY std::execution::seq
#elif defined(LAMBDA_LANCZOS_STDPAR_PAR)
#define LAMBDA_LANCZOS_STDPAR_POLICY std::execution::par
#elif defined(LAMBDA_LANCZOS_STDPAR_PAR_UNSEQ)
#define LAMBDA_LANCZOS_STDPAR_POLICY std::execution::par_unseq
#elif defined(LAMBDA_LANCZOS_STDPAR_UNSEQ)
#define LAMBDA_LANCZOS_STDPAR_POLICY std::execution::unseq
#endif

#endif /* LAMBDA_LANCZOS_MACRO_H_ */