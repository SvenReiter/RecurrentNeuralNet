
import java.util.ArrayList;
import java.util.List;

public class NeuralNet {

    private final List<List<Edge>> edges;
    private final List<List<Perceptron>> perceptronsPerLayer;

    /**
     * @param layersPerceptrons index describes nr of layer, value describes nr of perceptrons in.
     * @pre <layersPerceptrons> needs to contain at least 2 elements(input-/output-layer).
     * @post global <perceptronsPerLayer> got defined.
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

        List<List<Edge>> edges= new ArrayList<>();
        for(int l=ZERO; l< layersPerceptrons.length; ++l) {
            edges.add(new ArrayList<>());
            for(int f=ZERO; f< layersPerceptrons[l]; ++f) {
                if((l+ONE)<layersPerceptrons.length) {
                    for (int t=ZERO; t< layersPerceptrons[l+ONE]; ++t) {
                        edges.get(l).add(new Edge(perceptronsPerLayer.get(l).get(f), perceptronsPerLayer.get(l+ONE).get(t)));
                    }
                }
            }
        }
        this.edges= edges;
    }

    /**
     * for testing
     * @param edges  To not enter global address-room, on referencing the edges.
     * @param perceptronsPerLayer To not enter global address-room, on referencing the perceptrons. */
    public NeuralNet(List<List<Edge>> edges, List<List<Perceptron>> perceptronsPerLayer) {
        this.edges= edges;
        this.perceptronsPerLayer= perceptronsPerLayer;
    }

    /**
     * @param activationFunc The activation function to be applied to perceptrons.
     * @param edges To not enter global address-room, on referencing the edges.
     * @param input The input values for input layers <Perceptron.sum>s.
     * @post every <Perceptron.output> got defined. */
    public void forward(Func activationFunc, List<List<Edge>> edges,
                        List<List<Perceptron>> perceptronsPerLayer, List<Float> input) {
        final byte ZERO= 0;
        for(List<Edge> l: edges) {
            for (Edge e : l) {
                e.to.sum = ZERO;
                e.to.output = ZERO;
            }
        }

        for(int i=ZERO; i<input.size(); ++i) {
            if(!perceptronsPerLayer.get(ZERO).get(i).isBias) {
                perceptronsPerLayer.get(ZERO).get(i).sum = input.get(i);
                perceptronsPerLayer.get(ZERO).get(i).output = activationFunc.calc(perceptronsPerLayer.get(ZERO).get(i).sum);
            }
        }

        for(List<Edge> l: edges) {
            for (Edge e : l) {
                e.to.sum+= e.from.output* e.weight;
            }
            for (Edge e : l) {
                if(Float.compare(e.to.output, ZERO)== 0) {
                    e.to.output= activationFunc.calc(e.to.sum);
                }
            }
        }
    }

    /**
     * @param lossFunc The loss function to measure the fault of the predictions and the actual outputs.
     * @param edges To not enter global address-room, on referencing the edges.
     * @param target The targeting values for output layers <Perceptron.sum>s.  */
    public void backpropagate(Func sigDerivation, Func deltaLearning, Func lossFunc, List<Perceptron> output,
                              List<List<Edge>> edges, List<Float> target) {
        //Calc outputlayers <delta>
        for(int i=0; i< output.size(); ++i) {
            output.get(i).delta= sigDerivation.calc(output.get(i).sum)* lossFunc.calc(target.get(i), output.get(i).output);
        }

        //Remembers last edgesLayer ID
        int LAYER_ID= edges.size()-1;

        perceptronsPerLayer.get(LAYER_ID).get(1).delta=
            sigDerivation.calc(perceptronsPerLayer.get(LAYER_ID).get(1).sum)*
                (
                    (edges.get(LAYER_ID).get(1).to.delta* edges.get(LAYER_ID).get(1).weight) +
                    (edges.get(LAYER_ID).get(4).to.delta* edges.get(LAYER_ID).get(4).weight)
                );

        perceptronsPerLayer.get(LAYER_ID).get(2).delta=
            sigDerivation.calc(perceptronsPerLayer.get(LAYER_ID).get(2).sum)*
                (
                    (edges.get(LAYER_ID).get(2).to.delta* edges.get(LAYER_ID).get(2).weight) +
                    (edges.get(LAYER_ID).get(5).to.delta* edges.get(LAYER_ID).get(5).weight)
                );

        edges.get(LAYER_ID).get(0).weight+= deltaLearning.calc(edges.get(LAYER_ID).get(0).to.delta, edges.get(LAYER_ID).get(0).from.output);
        edges.get(LAYER_ID).get(1).weight+= deltaLearning.calc(edges.get(LAYER_ID).get(1).to.delta, edges.get(LAYER_ID).get(1).from.output);
        edges.get(LAYER_ID).get(2).weight+= deltaLearning.calc(edges.get(LAYER_ID).get(2).to.delta, edges.get(LAYER_ID).get(2).from.output);
        edges.get(LAYER_ID).get(3).weight+= deltaLearning.calc(edges.get(LAYER_ID).get(3).to.delta, edges.get(LAYER_ID).get(3).from.output);
        edges.get(LAYER_ID).get(4).weight+= deltaLearning.calc(edges.get(LAYER_ID).get(4).to.delta, edges.get(LAYER_ID).get(4).from.output);
        edges.get(LAYER_ID).get(5).weight+= deltaLearning.calc(edges.get(LAYER_ID).get(5).to.delta, edges.get(LAYER_ID).get(5).from.output);

        LAYER_ID--;

        edges.get(LAYER_ID).get(0).weight+= deltaLearning.calc(edges.get(LAYER_ID).get(0).to.delta, edges.get(LAYER_ID).get(0).from.output);
        edges.get(LAYER_ID).get(1).weight+= deltaLearning.calc(edges.get(LAYER_ID).get(1).to.delta, edges.get(LAYER_ID).get(1).from.output);
        edges.get(LAYER_ID).get(2).weight+= deltaLearning.calc(edges.get(LAYER_ID).get(2).to.delta, edges.get(LAYER_ID).get(2).from.output);
        edges.get(LAYER_ID).get(3).weight+= deltaLearning.calc(edges.get(LAYER_ID).get(3).to.delta, edges.get(LAYER_ID).get(3).from.output);
        edges.get(LAYER_ID).get(4).weight+= deltaLearning.calc(edges.get(LAYER_ID).get(4).to.delta, edges.get(LAYER_ID).get(4).from.output);
        edges.get(LAYER_ID).get(5).weight+= deltaLearning.calc(edges.get(LAYER_ID).get(5).to.delta, edges.get(LAYER_ID).get(5).from.output);
    }

    /**
     * @param epsilon
     * @param out
     * @param in
     * @post defines specific out.delta */
    private void calcDeltaLearning(float epsilon, Perceptron out, Perceptron in) {
        out.delta= epsilon* out.delta* in.output;
    }

    public List<List<Edge>> getEdges() {
        return edges;
    }
    public List<List<Perceptron>> getPerceptronsPerLayer() {
        return perceptronsPerLayer;
    }
}