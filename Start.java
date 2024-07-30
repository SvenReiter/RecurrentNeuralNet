import java.util.Arrays;

public class Start {
    public static void main(String[] args) {
        /*
         *  lineare Regression: Spezialfall der Regressionsanalyse, mit dem versucht wird eine beobachtete
         *                      abhängige Variable durch eine oder mehrere unabhängige Variablen zu erklären.
         *                      Bei der linearen Regression wird dabei ein lineares Modell angenommen.
         *  Aktivierungsfunktion: Sie macht aus einem linearen Wert einen nicht linearen Wert
         *  rekurrentes Netz: Rückgekoppeltes Netz
         *  Gradientenabstiegsverfahren:    Gradient:= bezeichnet den Verlauf der Änderung einer bestimmten Größe.
         *                                  Eine Optimierungsmethode um das Minimum einer Funktion zu finden. Indem es
         *                                  iterativ in die Richtung des steilsten Abstiegs der Funktion fortschreitet.
         *                                  Mit Ziel die Kostenfunktion/Verlustfunktion zu minimieren.
         *                                  Die Schrittweite/Lernrate/Eta bestimmt die Schrittgröße in Richtung des
         *                                  steilsten Abstiegs.
         */
        NeuralNet nn= new NeuralNet(4,3,3,3);

        Func activationFunc= x-> (float)(1 / (1 + Math.exp(-x[0])));
        nn.forward( activationFunc,
                    nn.getEdges(),
                    nn.getPerceptrons(),
                    Arrays.asList(0.1f, 0.2f, 0.3f, 0.4f));

        Func lossFunc= (x)-> (float)Math.pow((x[0]-x[1]), 2);
        Func sig = x-> (float)(1 / (1 + Math.exp(-x[0])));
        Func sigDx = x-> (sig.calc(x)* (1-sig.calc(x)));
        Func deltaLearning= x-> (x[0]* x[1]* x[2]);
        nn.backpropagate(   sigDx, deltaLearning, lossFunc,
                            nn.getPerceptrons().get(nn.getPerceptrons().size()-1),
                            nn.getEdges(),
                            Arrays.asList(0.4f, 0.3f, 0.2f, 0.1f));
    }
}