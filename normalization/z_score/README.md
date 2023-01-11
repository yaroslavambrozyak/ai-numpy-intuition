# Z-score normalization
After z-score normalization, all features will have a mean of 0 and a standard deviation of 1.
## Formula
<img src="https://latex.codecogs.com/svg.latex?\Large&space;x^{(i)}_j=\dfrac{x^{(i)}_j-\mu_j}{\sigma_j}" /> <br>
Where  𝑗  selects a feature or a column in the  𝐗  matrix.  µ𝑗  is the mean of all the values for feature (j) and  𝜎𝑗  is the standard deviation of feature (j).<br>

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mu_j=\frac{1}{m}%20\sum_{i=0}^{m-1}%20x^{(i)}_j" /> <br>
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\sigma^2_j=%20\frac{1}{m}%20\sum_{i=0}^{m-1}%20(x^{(i)}_j%20-%20\mu_j)^2" /> <br>
