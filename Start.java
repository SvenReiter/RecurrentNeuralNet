import java.util.Arrays;

public class Start {
    public static void main(String[] args) {
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
        nn.backpropagate(   activationFunc,
                            nn.getEdges(),
                            Arrays.asList(0.4f, 0.3f, 0.2f, 0.1f));
        System.out.println(nn.getOutput(activationFunc));
    }
}
