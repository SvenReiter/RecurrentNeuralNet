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

    private List<Edge> edges;

    /**
     * @param layersPerceptrons sequence of ints, a value describes the amount of perceptrons,
     *                          the sequences values describes the amount of layers
     * @pre <layersPerceptrons> needs to contain at least 2 elements(input-/output-layer).
     * @post global <edges> got defined.
     * @throws IllegalArgumentException Network needs to contain at least 2 elements(input-/output-layer).*/
    public NeuralNet(int... layersPerceptrons)
            throws IllegalArgumentException {

        final byte TWO= 2;
        if(layersPerceptrons.length<TWO)
            throw new IllegalArgumentException("Network needs to contain at least 2 elements(input-/output-layer).");

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
        this.edges= edges;

        System.out.println("Neural Network got initialized");

    }

    /**
     *
     * @param activationFunc
     * @param edges */
    public void forwarding(Func activationFunc, List<Edge> edges, List<Float> input) {
        for(Edge e: edges) {
            e.to.sum= 0f;
        }
        System.out.println("Perceptrons sums got initialized");

        for(int i=0; i<input.size(); ++i) {
            edges.get(0).from.sum= input.get(i);
        }
        System.out.println("Input-Layer got initialized");

        for(Edge e: edges) {
            e.to.sum+= activationFunc.calc(e.from.sum) * e.weight;
        }
        System.out.println("Neural Network got forwarded");
    }

    public List<Edge> getEdges() {
        return edges;
    }

    /**
     *
     * @param lossFunc
     * @param edges
     * @param predicted */
    public void backpropagation(Func lossFunc, List<Edge> edges, List<Float> predicted) {
        //<lossFunc> wird in der Regel bei der Optimierung des Modells verwendet,
        // um die Gewichte und Schwellenwerte anzupassen und den Fehler zu minimieren.
        //todo
    }
}
