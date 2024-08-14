
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
        final byte TWO= (byte)2;
        if(layersPerceptrons.length<TWO)
            throw new IllegalArgumentException("Network needs to contain at least 2 elements(input-/output-layer).");

        final byte ONE= (byte)1, O= (byte)0;
        List<List<Perceptron>> perceptronsPerLayer= new ArrayList<>();
        for(int l=O; l< layersPerceptrons.length; ++l) {
            perceptronsPerLayer.add(new ArrayList<>());
            for(int p=O; p<layersPerceptrons[l]; ++p) {
                perceptronsPerLayer.get(l).add(new Perceptron());
            }
        }
        this.perceptronsPerLayer= perceptronsPerLayer;

        List<List<Edge>> edges= new ArrayList<>();
        for(int l=O; l< layersPerceptrons.length-1; ++l) {
            edges.add(new ArrayList<>());
            for(int f=O; f< layersPerceptrons[l]; ++f) {
                if((l+ONE)<layersPerceptrons.length) {
                    for (int t=O; t< layersPerceptrons[l+ONE]; ++t) {
                        edges.get(l).add(new Edge(  perceptronsPerLayer.get(l).get(f),
                                                    perceptronsPerLayer.get(l+ONE).get(t)));
                    }
                }
            }
        }
        this.edges= edges;
    }

    /**
     * @param edges  To not enter global address-room, on referencing the edges.
     * @param perceptronsPerLayer To not enter global address-room, on referencing the perceptrons. */
    public NeuralNet(List<List<Edge>> edges, List<List<Perceptron>> perceptronsPerLayer) {
        this.edges= edges;
        this.perceptronsPerLayer= perceptronsPerLayer;
    }

    /**
     * @post Bias-perceptron got added to every layer except output layer. */
    public void addBias() {
        final byte ONE= (byte)1, O= (byte)0;
        final Perceptron bias= new Perceptron();
        bias.input = ONE;
        bias.output= ONE;
        bias.isBias= true;
        for(int l=O; l<perceptronsPerLayer.size()-ONE; ++l) {
            perceptronsPerLayer.get(l).add(bias);
            for(Perceptron p: perceptronsPerLayer.get(l+ONE)) {
                if(!p.isBias) {
                    edges.get(l).add(new Edge(bias, p));
                }
            }
        }

        Perceptron toSet, toRemember;
        for(int l=O; l< perceptronsPerLayer.size()-ONE; l++) {
            toSet= perceptronsPerLayer.get(l).get(O);
            for(int i=O; i<perceptronsPerLayer.get(l).size()-ONE; ++i) {
                toRemember= perceptronsPerLayer.get(l).get(i+ONE);
                perceptronsPerLayer.get(l).set(i+ONE, toSet);
                toSet= toRemember;
            }
            perceptronsPerLayer.get(l).set(O, bias);
        }
    }

    /**
     * @param activationFunc The activation function to be applied to perceptrons.
     * @param edges To not enter global address-room, on referencing the edges.
     * @param input The input values for input layers <Perceptron.sum>s.
     * @post every <Perceptron.output> got defined. */
    public void forward(Func activationFunc, List<List<Edge>> edges,
                        List<List<Perceptron>> perceptronsPerLayer, List<Float> input) {
        final byte O= (byte)0;
        for(List<Edge> l: edges) {
            for (Edge e : l) {
                e.to.input = O;
                e.to.output = O;
            }
        }

        for(int i=O; i<input.size(); ++i) {
            if(!perceptronsPerLayer.get(O).get(i).isBias) {
                perceptronsPerLayer.get(O).get(i).input = input.get(i);
                perceptronsPerLayer.get(O).get(i).output = perceptronsPerLayer.get(O).get(i).input;
            }
        }

        for(List<Edge> l: edges) {
            for (Edge e : l) {
                e.to.input += e.from.output* e.weight;
            }
            for (Edge e : l) {
                if(Float.compare(e.to.output, O)== O) {
                    e.to.output= activationFunc.calc(e.to.input);
                }
            }
        }
    }

    /**
     * @param activationFuncDx The derivative of used activation function in forwarding.
     * @param learningRule To adjust <weight> regarding given <delta>
     * @param lossFunc The loss function to measure the fault of the predictions and the actual outputs.
     * @param output To not enter global address-room, on referencing the output-layers List<Perceptron>.
     * @param edges To not enter global address-room, on referencing the edges.
     * @param target The targeting values for output layers <Perceptron.sum>s.
     * @post edges weights have been refined to achieve <target> more clearly.
     * @throws IllegalArgumentException when Output-Layer's size doesn't match size of <target>. */
    public void backpropagate(Func activationFuncDx, Func learningRule, Func lossFunc, List<Perceptron> output,
                              List<List<Edge>> edges, List<Float> target) throws IllegalArgumentException {
        final byte O= (byte)0, ONE= (byte)1;

        if(output.size()!= target.size())
            throw new IllegalArgumentException("Output-Layer's size doesn't match size of <target>.");

        //Calc output-layer's <delta>
        for(int i=O; i< output.size(); ++i) {
            output.get(i).delta=
                    activationFuncDx.calc(output.get(i).input)* lossFunc.calc(target.get(i), output.get(i).output);
        }

        //Calc other layer's <delta>
        for(int i=edges.size()-ONE; i>(-ONE); --i) {
            for(Edge e: edges.get(i)) {
                e.weight+= learningRule.calc(e.to.delta, e.from.output);                                                //optimize <weight>
                e.from.delta+= e.to.delta* e.weight;
            }
            for(Perceptron p: perceptronsPerLayer.get(i)) {
                p.delta= activationFuncDx.calc(p.input)*p.delta;
            }
        }
    }

    /**
     * @return a sequence(layers) of sequence(perceptrons) */
    public List<List<Perceptron>> getPerceptrons() {
        return perceptronsPerLayer;
    }

    public List<List<Edge>> getEdges() {
        return edges;
    }
}