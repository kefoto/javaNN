package nn;

public class HardThresholdAF implements ActivationFunction{
	public double activation(double z) {
		return (z >= 0.0) ? 1.0 : 0.0;
	}

	public double derivative(double z) {
		return 0;
	}
}
