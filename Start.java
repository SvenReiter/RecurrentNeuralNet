import java.util.Arrays;
import java.util.List;

public class Start {
    public static void main(String[] args) {
        NeuralNet nn= new NeuralNet(4,3,3,3);
        /*
         *  RNN:= Als rekurrente bzw. rückgekoppelte neuronale Netze bezeichnet man neuronale Netze, die sich im
         *        Gegensatz zu den Feedforward-Netzen durch Verbindungen von Neuronen einer Schicht zu Neuronen
         *        derselben oder einer vorangegangenen Schicht auszeichnen.
         *  Regression:= Beziehungen zwischen einer abhängigen und einer/mehreren unabhängigen Variablen zu modellieren.
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
         *  CNN:= Grundsätzlich besteht die Struktur eines klassischen Convolutional (diskrete Faltung:= mathematischer
         *        Operator, der für zwei Funktionen f und g eine dritte Funktion f*g liefert; Convolutional-
         *        Operation:= Filtering Operation) Neural Networks aus einem oder mehreren
         *        Convolutional Layer(=Filter:= Feature-Detectors; Result: Feature-Map), gefolgt von einem
         *        Pooling(zusammenbringen; hier werden überflüssige Informationen verworfen; zB. pooling(4px->1px)
         *        Max(höchsten Wert der 4 Pixel abbilden)-/Average(Durchschnittswert der 4 Pixel abbilden)-
         *        pooling: Positions-Invariante Merkmalserkennung) -Layer.
         *        Angegeben werden muss die Anzahl/Größe der Filter die pro Convolutional-Layer eingesetzt werden soll.
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