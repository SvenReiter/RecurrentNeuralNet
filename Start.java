import java.util.Arrays;

public class Start {
    public static void main(String[] args) {
        /*
            Implements static calculation in NeuralNet().backpropagate;
            Implements tests for testing forward and backpropagate;
            Adds field <delta> to Perceptron();
            Separates NeuralNet()s <edges> to List<List<Edge> instead of List<Edge>;
            Defines every <Perceptron.output> in NeuralNet.forwarding() when calculated and
              reuses that value for upcoming calculations;
            Implements calcDelta(float, Perceptron, Perceptron) in NeuralNet();
            Adds flag <isBias> to Perceptron();
            Checks <isBias> when calculating Perceptron().sum/.output;
         */
        /*
         *  lineare Regression:= Spezialfall der Regressionsanalyse, mit dem versucht wird eine beobachtete
         *                       abhängige Variable durch eine oder mehrere unabhängige Variablen zu erklären.
         *                       Bei der linearen Regression wird dabei ein lineares Modell angenommen.
         *  Aktivierungsfunktion:= Sie macht aus einem linearen Wert einen nicht linearen Wert
         */
        NeuralNet nn= new NeuralNet(4,3,3,3);

        Func activationFunc= x-> (float)(1 / (1 + Math.exp(-x[0])));
        nn.forward( activationFunc,
                    nn.getEdges(),
                    nn.getPerceptronsPerLayer(),
                    Arrays.asList(0.1f, 0.2f, 0.3f, 0.4f));

        Func lossFunc= (x)-> (float)Math.pow((x[0]-x[1]), 2);
        Func sig = x-> (float)(1 / (1 + Math.exp(-x[0])));
        Func sigDx = x-> (sig.calc(x)* (1-sig.calc(x)));
        Func deltaLearning= x-> (x[0]* x[1]* x[2]);
        nn.backpropagate(   sigDx, deltaLearning, lossFunc,
                            nn.getPerceptronsPerLayer().get(nn.getPerceptronsPerLayer().size()-1),
                            nn.getEdges(),
                            Arrays.asList(0.4f, 0.3f, 0.2f, 0.1f));
    }
}