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

    private final List<Edge> edges;
    private final List<List<Perceptron>> perceptronsPerLayer;

    /**
     * @param layersPerceptrons index describes nr of layer, value describes nr of perceptrons in.
     * @pre <layersPerceptrons> needs to contain at least 2 elements(input-/output-layer).
     * @post global <edges> got defined.
     * @throws IllegalArgumentException Network needs to contain at least 2 elements(input-/output-layer).*/
    public NeuralNet(int... layersPerceptrons)
            throws IllegalArgumentException {

        final byte TWO= 2;
        if(layersPerceptrons.length<TWO)
            throw new IllegalArgumentException("Network needs to contain at least 2 elements(input-/output-layer).");

        final byte ONE= 1, ZERO= 0;
        List<List<Perceptron>> perceptronsPerLayer= new ArrayList<>();
        for(int l=ZERO; l< layersPerceptrons.length; ++l) {
            perceptronsPerLayer.add(new ArrayList<>());
            for(int p=ZERO; p<layersPerceptrons[l]; ++p) {
                perceptronsPerLayer.get(l).add(new Perceptron());
            }
        }
        this.perceptronsPerLayer= perceptronsPerLayer;
        List<Edge> edges= new ArrayList<>();
        for(int l=ZERO; l< layersPerceptrons.length; ++l) {
            for(int f=ZERO; f< layersPerceptrons[l]; ++f) {
                if((l+ONE)<layersPerceptrons.length) {
                    for (int t=ZERO; t< layersPerceptrons[l+ONE]; ++t) {
                        edges.add(new Edge(perceptronsPerLayer.get(l).get(f), perceptronsPerLayer.get(l+ONE).get(t)));
                    }
                }
            }
        }
        this.edges= edges;

        System.out.println("Neural Network got initialized");

    }

    /**
     * @param activationFunc The activation function to be applied to perceptrons.
     * @param edges To not enter global address-room, on referencing the edges.
     * @param input The input values for input layers <Perceptron.sum>s. */
    public void forward(Func activationFunc, List<Edge> edges,
                        List<List<Perceptron>> perceptronsPerLayer, List<Float> input) {
        final byte ZERO= 0;
        for(Edge e: edges) {
            e.to.sum= ZERO;
        }
        System.out.println("Perceptrons sums got initialized");

        for(int i=ZERO; i<input.size(); ++i) {
            perceptronsPerLayer.get(ZERO).get(i).sum= input.get(i);
        }
        System.out.println("Input-Layer got initialized");

        for(Edge e: edges) {
            e.to.sum+= activationFunc.calc(e.from.sum) * e.weight;
        }
        System.out.println("Neural Network got forwarded");
    }

    /**
     * @param lossFunc The loss function to measure the fault of the predictions and the actual outputs.
     * @param edges To not enter global address-room, on referencing the edges.
     * @param predicted The predicted values for output layers <Perceptron.sum>s.  */
    public void backpropagate(Func lossFunc, List<Edge> edges, List<Float> predicted) {
        //<lossFunc> wird in der Regel bei der Optimierung des Modells verwendet,
        // um die Gewichte und Schwellenwerte anzupassen und den Fehler zu minimieren.
        //todo
    }

    public List<Edge> getEdges() {
        return edges;
    }
    public List<List<Perceptron>> getPerceptronsPerLayer() {
        return perceptronsPerLayer;
    }
}
