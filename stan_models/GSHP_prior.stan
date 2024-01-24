/* Gaussian prior & measurement likelihood model */
data {
    real mu;
    real sigma;
    real error;
}
parameters {
    real<lower=1e-3> theta; // theta > 0 for valid variance
    real<lower=1e-3> z;
}
model {
    theta ~ normal(mu,sigma);
    z ~ normal(theta,error*theta);
}