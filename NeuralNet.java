
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
    public NeuralNet(int... layersPerceptrons) throws IllegalArgumentException {
        // TODO: 15.07.2024 Initialize a bias-perceptron to every layer except output-layer.
        final byte TWO= 2;
        if(layersPerceptrons.length<TWO)
            throw new IllegalArgumentException("Network needs to contain at least 2 elements(input-/output-layer).");

        final byte ONE= 1, O= 0;
        List<List<Perceptron>> perceptronsPerLayer= new ArrayList<>();
        for(int l=O; l< layersPerceptrons.length; ++l) {
            perceptronsPerLayer.add(new ArrayList<>());
            for(int p=O; p<layersPerceptrons[l]; ++p) {
                perceptronsPerLayer.get(l).add(new Perceptron());
            }
        }
        this.perceptronsPerLayer= perceptronsPerLayer;

        List<List<Edge>> edges= new ArrayList<>();
        for(int l=O; l< layersPerceptrons.length; ++l) {
            edges.add(new ArrayList<>());
            for(int f=O; f< layersPerceptrons[l]; ++f) {
                if((l+ONE)<layersPerceptrons.length) {
                    for (int t=O; t< layersPerceptrons[l+ONE]; ++t) {
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
        final byte O= 0;
        for(List<Edge> l: edges) {
            for (Edge e : l) {
                e.to.sum = O;
                e.to.output = O;
            }
        }

        for(int i=O; i<input.size(); ++i) {
            if(!perceptronsPerLayer.get(O).get(i).isBias) {
                perceptronsPerLayer.get(O).get(i).sum = input.get(i);
                perceptronsPerLayer.get(O).get(i).output = perceptronsPerLayer.get(O).get(i).sum;
            }
        }

        for(List<Edge> l: edges) {
            for (Edge e : l) {
                e.to.sum+= e.from.output* e.weight;
            }
            for (Edge e : l) {
                if(Float.compare(e.to.output, O)== O) {
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
        final byte O= 0, ONE= 1;
        //Calc outputlayers <delta>
        for(int i=O; i< output.size(); ++i) {
            output.get(i).delta= sigDerivation.calc(output.get(i).sum)* lossFunc.calc(target.get(i), output.get(i).output);
        }

        //Calc remaining layers <delta>
        for(int i=edges.size()-ONE; i>(-ONE); --i) {
            for(Edge e: edges.get(i)) {
                e.from.delta+= e.to.delta* e.weight;  //At first <delta> gets used to sum up (prevLayerDelta*weight)s
            }
            for(Perceptron p: perceptronsPerLayer.get(i)) {
                p.delta= sigDerivation.calc(p.sum)*p.delta;
            }
        }

        for(List<Edge> eL: edges) {
            for(Edge e: eL) {
                e.weight += deltaLearning.calc(e.to.delta, e.from.output);
            }
        }
    }

    public List<List<Edge>> getEdges() {
        return edges;
    }
    public List<List<Perceptron>> getPerceptronsPerLayer() {
        return perceptronsPerLayer;
    }
}