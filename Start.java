
public class Start {
    public static void main(String[] args) {
        NeuralNet nn= new NeuralNet(
                x-> (1 / (1 + Math.exp(-x[0]))),
                x-> (Math.pow((x[0]-x[1]), 2)),
                4,3,3,3
        );
    }
}
