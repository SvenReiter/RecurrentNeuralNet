import java.util.ArrayList;
import java.util.List;

public class NeuralNet {
    /*
    Neuronale Netzwerke bestehen dabei im Prinzip immer aus den folgenden vier Komponenten:

        - Einem Input Layer
        - Mindestens einem Hidden Layer
        - Mindestens einer Aktivierungsfunktion
        - Einem Output Layer

        - Um Werte zu summieren, muss kontextbasiert verarbeitet werden => <Perceptron>s müssen zur Laufzeit existieren!

     */

    /**
     * @param layersPerceptrons
     * @pre <layersPerceptrons> needs to contain at least 3 elements. */
    public NeuralNet(Func activationFunc, Func lossFunc, int... layersPerceptrons) {
        final byte ONE= 1, ZERO= 0;
        List<List<Perceptron>> perceptronLayers= new ArrayList<>();
        for(int l=ZERO; l< layersPerceptrons.length; ++l) {
            perceptronLayers.add(new ArrayList<>());
            for(int p=ZERO; p<layersPerceptrons[l]; ++p) {
                perceptronLayers.get(l).add(new Perceptron());
            }
        }

        List<Edge> edges= new ArrayList<>();
        for(int l=ZERO; l< layersPerceptrons.length; ++l) {
            for(int f=ZERO; f< layersPerceptrons[l]; ++f) {
                if((l+ONE)<layersPerceptrons.length) {
                    for (int t=ZERO; t< layersPerceptrons[l+ONE]; ++t) {
                        edges.add(new Edge(perceptronLayers.get(l).get(f), perceptronLayers.get(l+ONE).get(t)));
                    }
                }
            }
        }
        System.out.println("PerceptronLayers got initialized");
    }

    public void forwarding(Func activationFunc) {

    }

    public void backpropagation(Func lossFunc) {
        //<lossFunc> wird in der Regel bei der Optimierung des Modells verwendet,
        // um die Gewichte und Schwellenwerte anzupassen und den Fehler zu minimieren.
    }
}
