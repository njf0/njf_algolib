import matplotlib.pyplot as plt
import numpy as np

class PolynomialRegression(object):
    """
    Estimate coefficients of y = b_0 + b_1*x + b_2*x^2 + ... + b_i*x^i
    for a dataset through least-squares approximation.
    
    Object oriented approach not really necessary but I always find it useful to practice.

    Parameters
    ----------
    order: int
        Order of the polynomial to be fitted to data.

    x_lims: tuple
        x-axis limits of the data, for example: (-10, 10).
    """

    def __init__(self,
                 order: int,
                 x_lims: tuple) -> None:
        
        self.order = order
        self.x_lims = x_lims

        return None

    def polynomial(self,
                   x: np.ndarray,
                   coeffs: np.ndarray) -> np.ndarray:
        """
        Evaluate a polynomial of order len(coeffs) on x.

        Parameters
        ----------
        x: np.ndarray
            An array of points on the x-axis over which to evaluate the polynomial.

        coeffs: np.ndarray
            Array of polynomial coefficients in increasing order.

        Returns
        -------
        y: np.ndarray
            Array of evaluated points.
        """
        y = [sum([c * np.power(i, o) for o, c in enumerate(coeffs)]) for i in x]

        return np.array(y)

    def generate_data(self,
                      coeffs: np.ndarray,
                      add_noise: bool = False) -> np.ndarray:
        """
        Generates noisy y = f(x) data where f(x) has order len(coeffs).

        Parameters
        ----------
        coeffs: np.ndarray
            Array of polynomial coefficients in increasing order.

        add_noise: bool
            Choose whether or not to add Gaussian noise to data.
            
        Returns
        -------
        x: np.ndarray
            Array of x-axis data points.

        y: np.ndarray
            Array of y-axis data points.
        """
        x = np.linspace(self.x_lims[0], self.x_lims[1], 101)
        y = self.polynomial(x, coeffs)
        
        if add_noise:
            y = y + (self.x_lims[1] - self.x_lims[0]) * np.random.normal(size = y.shape[0])
            
        return x, y

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> np.ndarray:
        """
        Discovers coefficients of n-th degree polynomial

        Parameters
        ----------
        x: np.ndarray
            Array of x-axis data points.

        y: bool
            Array of y-axis data points.
            
        Returns
        -------
        theta: np.ndarray
            List of coefficients in increasing order.
        """
        X = np.transpose(np.array([np.power(x, i) for i in range(self.order + 1)]))
        X_T = np.transpose(X)
        X_TX = np.dot(X_T, X)
        X_TX_inv = np.linalg.inv(X_TX)
        X_Ty = np.dot(X_T, y)
        theta = np.dot(X_TX_inv, X_Ty)
        
        return theta

    def mae(self,
            y_predic: np.ndarray,
            y_actual: np.ndarray) -> float:
        """
        Discovers coefficients of n-th degree polynomial

        Parameters
        ----------
        y_predic: np.ndarray
            Array of fitted y-axis data points.

        y_actual: np.ndarray
            Array of actual y-axis data points.
            
        Returns
        -------
        mae: float
            Mean absolute error of fit.
        """
        mae = sum([abs(y[1] - y[0]) for y in zip(y_predic, y_actual)]) / len(y_actual)

        return mae

if __name__ == '__main__':

    # Demonstration
    coeffs = [100, 2, -1, 1]
    print("Actual coefficients: {}".format(", ".join([str(i) for i in coeffs])))
    a = PolynomialRegression(order = 3, x_lims = (-10, 10))
    x, y = a.generate_data(coeffs, add_noise = True)
    theta = a.fit(x, y)
    print("Estimated coefficients: {}".format(", ".join([str(round(i, 2)) for i in theta])))
    y_predicted = a.polynomial(x, theta)
    mae = a.mae(y_predicted, y)
    print("Mean Absolute Error: {:.3f}.".format(mae))
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, 'x', label = "Data")
    ax.plot(x, y_predicted, label = "Fit")
    plt.axhline(y = 0, color = 'k')
    plt.axvline(x = 0, color = 'k')
    plt.legend(loc = 'best')
    plt.show()
