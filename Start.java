import java.util.Arrays;

public class Start {
    public static void main(String[] args) {
        /*
            Changes NeuralNet.edges to final because <edges> gets only defined in constructor;
            Adds global <perceptronsPerLayer> field and its get() to NeuralNet;
            Adds Param <perceptronsPerLayer> to NeuralNet.forwarding()s paramlist;
            Iterates <perceptronsPerLayer> on defining input-perceptrons instead of <edges> in NeuralNet.forwarding();
            Renames forwarding->forward, backpropagation->backpropagate in consequences of codequality;
            Removes unnecessary initialization/Initializer of <Perceptron.sum>;
         */
        /*
         *  lineare Regression:= Spezialfall der Regressionsanalyse, mit dem versucht wird eine beobachtete
         *                       abhängige Variable durch eine oder mehrere unabhängige Variablen zu erklären.
         *                       Bei der linearen Regression wird dabei ein lineares Modell angenommen.
         *  Aktivierungsfunktion:= Sie macht aus einem linearen Wert einen nicht linearen Wert
         */
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