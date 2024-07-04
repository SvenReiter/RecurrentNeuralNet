import java.util.Arrays;

public class Start {
    public static void main(String[] args) {
        NeuralNet nn= new NeuralNet(4,3,3,3);

        // TODO: review forwarding on runtime
        nn.forward(x-> (float)(1 / (1 + Math.exp(-x[0]))),
                        nn.getEdges(),
                        nn.getPerceptronsPerLayer(),
                        Arrays.asList(0.1f, 0.2f, 0.3f, 0.4f));
        nn.backpropagate(x-> (float)(Math.pow((x[0]-x[1]), 2)),
                            nn.getEdges(),
                            Arrays.asList(0.4f, 0.3f, 0.2f, 0.1f));

    }
}
