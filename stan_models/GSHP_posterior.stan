/* Gaussian posterior model given sample measurement */
data {
    real mu;
    real sigma;
    real error;
    real z;
}
parameters {
    real<lower=1e-3> theta; // theta > 0 for valid variance
}
model {
    theta ~ normal(mu,sigma);
    z ~ normal(theta,error*theta);
}