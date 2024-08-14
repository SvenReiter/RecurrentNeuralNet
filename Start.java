import java.util.Arrays;
import java.util.List;

public class Start {
    public static void main(String[] args) {
        NeuralNet nn= new NeuralNet(4,3,3,3);
        /*
         *  Rekurrentes Netz:= Rückgekoppeltes Netz
         *  Lineare Regression: Spezialfall der Regressionsanalyse, mit dem versucht wird eine beobachtete
         *                      abhängige Variable durch eine oder mehrere unabhängige Variablen zu erklären.
         *                      Bei der linearen Regression wird dabei ein lineares Modell angenommen.
         *  Aktivierungsfunktion: Damit ein Neuron basierend auf seinen Eingängen aktiviert oder deaktiviert wird.
         *                        Sie macht aus einem linearen Wert einen nicht linearen Wert und als Ergebnis
         *                        resultiert ein Wert zwischen 0 und 1.
         *                        Weiter kann einen Schwellenwert definiert sein welcher Aussage treffen kann,
         *                        wann ein Neuron aktiv wird bzw. sein Signal an folgende Perzeptron weitergibt.
         *  Kostenfunktion/Verlustfunktion: Sie wird als Fehler bezeichnet und spiegelt den Unterschied zwischen
         *                                  der Vorhersage des Netzes und dem Ziel wider.
         *  Lernregel: Regel um die Gewichte im Verlauf anzupassen.
         *  Gradientenabstiegsverfahren: Gradient:= bezeichnet den Verlauf der Änderung einer bestimmten Größe.
         *                               Eine Optimierungsmethode um das Minimum einer Funktion zu finden. Indem es
         *                               iterativ in die Richtung des steilsten Abstiegs der Funktion fortschreitet.
         *                               Mit Ziel die Kostenfunktion/Verlustfunktion zu minimieren.
         *                               Die Schrittweite/Lernrate/Eta bestimmt die Schrittgröße in Richtung des
         *                               steilsten Abstiegs.
         */
        Func activationFunc= x-> (float)(1 / (1 + Math.exp(-x[0])));
        nn.forward( activationFunc,
                    nn.getEdges(),
                    nn.getPerceptrons(),
                    Arrays.asList(0.1f, 0.2f, 0.3f, 0.4f));
        Func lossFunc= x-> (x[0]-x[1]);
        List<Float> target= Arrays.asList(0.3f, 0.2f, 0.1f);
        System.out.println("Fehler initial");
        for(int i=0; i<nn.getPerceptrons().get(nn.getPerceptrons().size()-1).size(); i++) {
            System.out.println("\t"+lossFunc.calc(target.get(i),
                                nn.getPerceptrons().get(nn.getPerceptrons().size()-1).get(i).output));
        }

        Func sig = x-> (float)(1 / (1 + Math.exp(-x[0])));
        Func sigDx = x-> (sig.calc(x)* (1-sig.calc(x)));
        Func deltaLearning= x-> (x[0]* x[1]);
        nn.backpropagate(   sigDx, deltaLearning, lossFunc,
                            nn.getPerceptrons().get(nn.getPerceptrons().size()-1),
                            nn.getEdges(),
                            target);
        nn.forward( activationFunc,
                    nn.getEdges(),
                    nn.getPerceptrons(),
                    Arrays.asList(0.1f, 0.2f, 0.3f, 0.4f));

        System.out.println("Fehler nach backpropagation()");
        for(int i=0; i<nn.getPerceptrons().get(nn.getPerceptrons().size()-1).size(); i++) {
            System.out.println("\t"+lossFunc.calc(target.get(i),
                                nn.getPerceptrons().get(nn.getPerceptrons().size()-1).get(i).output));
        }
    }
}