data {
  int<lower=0> num_msoas;        // Number of MSOAs
  int<lower=0> N;                // Total number of observations
  int<lower=0> K;                // Number of predictors
  int<lower=0> y[N];             // Crime counts
  matrix[N, K] X;                // Predictor matrix
  matrix[num_msoas, num_msoas] W; // Spatial weight matrix
  int<lower=0> num_years;        // Number of years
}

parameters {
  vector[K] beta;                // Coefficients for predictors
  real alpha;                    // Intercept
  vector[num_msoas] phi;         // Spatial random effects (one per MSOA)
  real<lower=0> sigma_phi;       // SD of spatial effects
}

model {
  matrix[num_msoas, num_msoas] Q;
  vector[num_msoas] D_diag;
  
  // Compute diagonal elements for matrix D
  for (i in 1:num_msoas) {
    D_diag[i] = sum(W[i, ]);
  }
  
  // Construct precision matrix Q = D - W, where D is the diagonal matrix of row sums of W
  Q = diag_matrix(D_diag) - W + diag_matrix(rep_vector(1e-5, num_msoas)); // Adding small value for numerical stability
  
  // Adjusted Priors
  beta ~ normal(0, 2);           // Narrowed prior for coefficients
  alpha ~ normal(0, 5);          // Narrowed prior for intercept
  sigma_phi ~ normal(0, 1);      // More informative prior for SD of spatial effects
  
  // Spatial effects prior using the precision matrix
  phi ~ multi_normal_prec(rep_vector(0, num_msoas), Q / sigma_phi^2);
  
  // Likelihood
  for (n in 1:N) {
    int msoa_index = (n - 1) %/% num_years + 1;  // Determine the MSOA index for observation n
    y[n] ~ poisson_log(alpha + dot_product(X[n], beta) + phi[msoa_index]);
  }
}

generated quantities {
  vector[N] y_pred;
  for (n in 1:N) {
    int msoa_index = (n - 1) %/% num_years + 1;  // Determine the MSOA index for prediction
    y_pred[n] = poisson_log_rng(alpha + dot_product(X[n], beta) + phi[msoa_index]);
  }
}

