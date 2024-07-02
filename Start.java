import java.util.Arrays;

public class Start {
    public static void main(String[] args) {
        /*
            Adds global List<Edge> edges to <NeuralNet> to hand over the control for e.g. forwarding() externally;
            Moves Perceptron.sum definition to initializer;
            Changes Edges.weight's range of value to (-1) up to 1(otherwise weights cannot get reduced anymore);
         */
        NeuralNet nn= new NeuralNet(4,3,3,3);
        nn.forwarding(x-> (float)(1 / (1 + Math.exp(-x[0]))), nn.getEdges(), Arrays.asList(0.1f, 0.2f, 0.3f, 0.4f));
        nn.backpropagation(x-> (float)(Math.pow((x[0]-x[1]), 2)), nn.getEdges(), Arrays.asList(0.4f, 0.3f, 0.2f, 0.1f));

    }
}