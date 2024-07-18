import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TestNeuralNet {
    final double DELTA= 1e-5;
    final float ETA= 1f;
    final Func sig = x-> (float)(1 / (1 + Math.exp(-x[0])));
    final Func lossFunc= x-> (x[0]-x[1]);
    final Func sigDx = x-> (sig.calc(x)* (1-sig.calc(x)));
    final Func deltaLearning= x-> (ETA* x[0]* x[1]);

    @Test
    void forwardTestA() {
        List<List<Perceptron>> perceptronsPerLayer= new ArrayList<>();
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(1).add(new Perceptron());

        List<List<Edge>> edges= new ArrayList<>();
        edges.add(new ArrayList<>());
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(0), perceptronsPerLayer.get(1).get(0), 0.7f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(1), perceptronsPerLayer.get(1).get(0), 0.8f));

        NeuralNet nn= new NeuralNet(edges, perceptronsPerLayer);
        nn.forward(x-> (float)(1 / (1 + Math.exp(-x[0]))),
                nn.getEdges(),
                nn.getPerceptrons(),
                Arrays.asList(0.5f, 0.6f));

        assertEquals(nn.getPerceptrons().get(1).get(0).sum, 0.83, DELTA);
    }

    @Test
    void forwardTestB() {
        List<List<Perceptron>> perceptronsPerLayer= new ArrayList<>();
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(1).add(new Perceptron());
        perceptronsPerLayer.get(1).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(2).add(new Perceptron());
        perceptronsPerLayer.get(2).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(3).add(new Perceptron());

        List<List<Edge>> edges= new ArrayList<>();
        edges.add(new ArrayList<>());
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(0), perceptronsPerLayer.get(1).get(0), 0.9f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(1), perceptronsPerLayer.get(1).get(0), 0.5f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(2), perceptronsPerLayer.get(1).get(0), 0.7f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(0), perceptronsPerLayer.get(1).get(1), 0.2f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(1), perceptronsPerLayer.get(1).get(1), 0.6f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(2), perceptronsPerLayer.get(1).get(1), 0.8f));

        edges.add(new ArrayList<>());
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(0), perceptronsPerLayer.get(2).get(0), 0.4f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(1), perceptronsPerLayer.get(2).get(0), 0.3f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(0), perceptronsPerLayer.get(2).get(1), 0.4f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(1), perceptronsPerLayer.get(2).get(1), 0.1f));

        edges.add(new ArrayList<>());
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(0), perceptronsPerLayer.get(3).get(0), 0.7f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(1), perceptronsPerLayer.get(3).get(0), 0.2f));

        NeuralNet nn= new NeuralNet(edges, perceptronsPerLayer);
        nn.forward(sig,
                nn.getEdges(),
                nn.getPerceptrons(),
                Arrays.asList(0.2f, 0.3f, 0.4f));

        assertEquals(nn.getPerceptrons().get(1).get(0).sum, 0.61, DELTA);
        assertEquals(nn.getPerceptrons().get(1).get(1).sum, 0.54, DELTA);

        assertEquals(nn.getPerceptrons().get(2).get(0).sum, 0.44872004, DELTA);
        assertEquals(nn.getPerceptrons().get(2).get(1).sum, 0.32235756, DELTA);

        assertEquals(nn.getPerceptrons().get(3).get(0).sum, 0.54321417, DELTA);

        assertEquals(nn.getPerceptrons().get(3).get(0).output, 0.6325598, DELTA);
    }

    @Test
    void forwardBackpropagateTestA() {
        List<List<Perceptron>> perceptronsPerLayer = new ArrayList<>();
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).get(0).sum= 1f;
        perceptronsPerLayer.get(0).get(0).output= 1f;
        perceptronsPerLayer.get(0).get(0).isBias= true;
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(1).add(new Perceptron());
        perceptronsPerLayer.get(1).get(0).sum= 1f;
        perceptronsPerLayer.get(1).get(0).output= 1f;
        perceptronsPerLayer.get(1).get(0).isBias= true;
        perceptronsPerLayer.get(1).add(new Perceptron());
        perceptronsPerLayer.get(1).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(2).add(new Perceptron());
        perceptronsPerLayer.get(2).add(new Perceptron());

        List<List<Edge>> edges= new ArrayList<>();
        edges.add(new ArrayList<>());
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(0), perceptronsPerLayer.get(1).get(1), 0.3f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(1), perceptronsPerLayer.get(1).get(1), 0.8f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(2), perceptronsPerLayer.get(1).get(1), 0.5f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(0), perceptronsPerLayer.get(1).get(2), -0.2f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(1), perceptronsPerLayer.get(1).get(2), -0.6f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(2), perceptronsPerLayer.get(1).get(2), 0.7f));
        edges.add(new ArrayList<>());
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(0), perceptronsPerLayer.get(2).get(0), 0.2f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(1), perceptronsPerLayer.get(2).get(0), 0.4f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(2), perceptronsPerLayer.get(2).get(0), 0.3f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(0), perceptronsPerLayer.get(2).get(1), 0.1f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(1), perceptronsPerLayer.get(2).get(1), -0.4f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(2), perceptronsPerLayer.get(2).get(1), 0.9f));

        NeuralNet nn= new NeuralNet(edges, perceptronsPerLayer);
        nn.forward( sig,
                nn.getEdges(),
                nn.getPerceptrons(),
                Arrays.asList(1f, 0.7f, 0.6f));

        assertEquals(nn.getPerceptrons().get(2).get(0).output, 0.6546592, DELTA);
        assertEquals(nn.getPerceptrons().get(2).get(1).output, 0.5499865, DELTA);

        nn.backpropagate(sigDx, deltaLearning, lossFunc,
                nn.getPerceptrons().get(nn.getPerceptrons().size()-1),
                nn.getEdges(), Arrays.asList(0.9f, 0.2f));

        assertEquals(nn.getPerceptrons().get(2).get(0).delta, 0.0554667897522, DELTA);
        assertEquals(nn.getPerceptrons().get(2).get(1).delta, -0.0866221413016, DELTA);

        assertEquals(nn.getEdges().get(1).get(1).to.delta, 0.0554667897522, DELTA);
        assertEquals(nn.getEdges().get(1).get(5).to.delta, -0.0866221413016, DELTA);

        int LAYER_ID= nn.getEdges().size()-1;

        assertEquals(edges.get(LAYER_ID).get(0).weight, 0.2554667897522, DELTA);
        assertEquals(edges.get(LAYER_ID).get(1).weight, 0.44222286838, DELTA);
        assertEquals(edges.get(LAYER_ID).get(2).weight, 0.3249600555, DELTA);
        assertEquals(edges.get(LAYER_ID).get(3).weight, 0.01338586984, DELTA);
        assertEquals(edges.get(LAYER_ID).get(4).weight, -0.465948273, DELTA);
        assertEquals(edges.get(LAYER_ID).get(5).weight, 0.861, DELTA);
                                                                                              //f'(netzinput bzw. <sum>)*(Summe über Richtung Output verlaufende Kanten zu nächstem Layer(PerzeptronDelta*KantenGewicht))
                                                                                              //bei der Rechnung müssen die ursprünglichen Gewichte verwendet werden
        assertEquals(nn.getEdges().get(1).get(1).from.delta, 0.01032732018, DELTA);     // = f'(1.1600001) * (0.0554667897522*0.4+(-0.0866221413016*(-0.4))
        assertEquals(nn.getEdges().get(1).get(2).from.delta, -0.01517768905, DELTA);    // = f'(-0.19999999) * (0.0554667897522*0.3+(-0.0866221413016*(0.9))

        //Delta-Lernregel   ->  deltaLearning: weight+= (<ETA>(1) * <to.delta> * <from.output>)
        LAYER_ID--;
        assertEquals(edges.get(LAYER_ID).get(0).weight, 0.3+0.01032732018, DELTA);  //0.3+= <ETA>*<toDelta>*<fromOutput>= (1)*0.01032732018*1(bias)
        assertEquals(edges.get(LAYER_ID).get(1).weight, 0.8+0.007229124, DELTA);   //0.8+= <ETA>*<toDelta>*<fromOutput>= (1)*0.01032732*0.7
        assertEquals(edges.get(LAYER_ID).get(2).weight, 0.5+0.006196392, DELTA);    //0.5+= <ETA>*<toDelta>*<fromOutput>= (1)*0.01032732*0.6
        assertEquals(edges.get(LAYER_ID).get(3).weight, -0.2-0.01517768905, DELTA);   //-0.2+= <ETA>*<toDelta>*<fromOutput>= (1)*-0.01517768905*1(bias)
        assertEquals(edges.get(LAYER_ID).get(4).weight, -0.6-0.01062438233, DELTA);   //-0.6+= <ETA>*<toDelta>*<fromOutput>= (1)*-0.01517768905*0.7
        assertEquals(edges.get(LAYER_ID).get(5).weight, 0.7-0.00910661343, DELTA);    //0.7+= <ETA>*<toDelta>*<fromOutput>= (1)*-0.01517768905*0.6
    }

    @Test
    void forwardBackpropagateTestB() {
        List<List<Perceptron>> perceptronsPerLayer = new ArrayList<>();
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).get(0).sum= 1f;
        perceptronsPerLayer.get(0).get(0).output= 1f;
        perceptronsPerLayer.get(0).get(0).isBias= true;
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(1).add(new Perceptron());
        perceptronsPerLayer.get(1).get(0).sum= 1f;
        perceptronsPerLayer.get(1).get(0).output= 1f;
        perceptronsPerLayer.get(1).get(0).isBias= true;
        perceptronsPerLayer.get(1).add(new Perceptron());
        perceptronsPerLayer.get(1).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(2).add(new Perceptron());
        perceptronsPerLayer.get(2).add(new Perceptron());

        List<List<Edge>> edges= new ArrayList<>();
        edges.add(new ArrayList<>());
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(0), perceptronsPerLayer.get(1).get(1), 0.3f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(1), perceptronsPerLayer.get(1).get(1), 0.8f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(2), perceptronsPerLayer.get(1).get(1), 0.5f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(0), perceptronsPerLayer.get(1).get(2), -0.2f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(1), perceptronsPerLayer.get(1).get(2), -0.6f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(2), perceptronsPerLayer.get(1).get(2), 0.7f));
        edges.add(new ArrayList<>());
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(0), perceptronsPerLayer.get(2).get(0), 0.2f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(1), perceptronsPerLayer.get(2).get(0), 0.4f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(2), perceptronsPerLayer.get(2).get(0), 0.3f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(0), perceptronsPerLayer.get(2).get(1), 0.1f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(1), perceptronsPerLayer.get(2).get(1), -0.4f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(2), perceptronsPerLayer.get(2).get(1), 0.9f));

        NeuralNet nn= new NeuralNet(edges, perceptronsPerLayer);
        nn.forward( sig,
                nn.getEdges(),
                nn.getPerceptrons(),
                Arrays.asList(1f, 0.7f, 0.6f));

        final float preResult_hA= nn.getPerceptrons().get(2).get(0).output;
        final float preResult_hB= nn.getPerceptrons().get(2).get(1).output;
        final float TARGET_oA= 0.9f, TARGET_oB= 0.2f;
        nn.backpropagate(sigDx, deltaLearning, lossFunc,
                nn.getPerceptrons().get(nn.getPerceptrons().size()-1),
                nn.getEdges(), Arrays.asList(TARGET_oA, TARGET_oB));

        nn.forward( sig,
                nn.getEdges(),
                nn.getPerceptrons(),
                Arrays.asList(1f, 0.7f, 0.6f));

        // tests whether trained output is closer to target than previous output
        Assertions.assertTrue(
    Math.abs(TARGET_oA-preResult_hA)>Math.abs(TARGET_oA-nn.getPerceptrons().get(2).get(0).output) &&
            Math.abs(TARGET_oB-preResult_hB)>Math.abs(TARGET_oB-nn.getPerceptrons().get(2).get(1).output));
    }

    @Test
    void forwardBackpropagateTestC() {
        List<List<Perceptron>> perceptronsPerLayer = new ArrayList<>();
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(1).add(new Perceptron());
        perceptronsPerLayer.get(1).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(2).add(new Perceptron());
        perceptronsPerLayer.get(2).add(new Perceptron());
        perceptronsPerLayer.get(2).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(3).add(new Perceptron());
        perceptronsPerLayer.get(3).add(new Perceptron());
        perceptronsPerLayer.get(3).add(new Perceptron());
        perceptronsPerLayer.get(3).add(new Perceptron());

        List<List<Edge>> edges= new ArrayList<>();
        edges.add(new ArrayList<>());
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(0), perceptronsPerLayer.get(1).get(0), 0.1f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(0), perceptronsPerLayer.get(1).get(1), 0.2f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(1), perceptronsPerLayer.get(1).get(0), 0.1f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(1), perceptronsPerLayer.get(1).get(1), 0.2f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(2), perceptronsPerLayer.get(1).get(0), 0.1f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(2), perceptronsPerLayer.get(1).get(1), 0.3f));
        edges.add(new ArrayList<>());
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(0), perceptronsPerLayer.get(2).get(0), 0.4f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(0), perceptronsPerLayer.get(2).get(1), 0.5f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(0), perceptronsPerLayer.get(2).get(2), 0.3f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(1), perceptronsPerLayer.get(2).get(0), 0.1f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(1), perceptronsPerLayer.get(2).get(1), 0.2f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(1), perceptronsPerLayer.get(2).get(2), 0.6f));
        edges.add(new ArrayList<>());
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(0), perceptronsPerLayer.get(3).get(0), 0.1f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(0), perceptronsPerLayer.get(3).get(1), 0.2f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(0), perceptronsPerLayer.get(3).get(2), 0.4f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(0), perceptronsPerLayer.get(3).get(3), 0.1f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(1), perceptronsPerLayer.get(3).get(0), 0.3f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(1), perceptronsPerLayer.get(3).get(1), 0.3f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(1), perceptronsPerLayer.get(3).get(2), 0.1f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(1), perceptronsPerLayer.get(3).get(3), 0.2f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(2), perceptronsPerLayer.get(3).get(0), 0.6f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(2), perceptronsPerLayer.get(3).get(1), 0.7f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(2), perceptronsPerLayer.get(3).get(2), 0.5f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(2), perceptronsPerLayer.get(3).get(3), 0.9f));

        NeuralNet nn= new NeuralNet(edges, perceptronsPerLayer);
        nn.forward( sig,
                    nn.getEdges(),
                    nn.getPerceptrons(),
                    Arrays.asList(0.5f, 0.7f, 0.9f));

        assertEquals(nn.getPerceptrons().get(3).get(0).output, 0.64928, DELTA);
        assertEquals(nn.getPerceptrons().get(3).get(1).output, 0.67615, DELTA);
        assertEquals(nn.getPerceptrons().get(3).get(2).output, 0.64658, DELTA);
        assertEquals(nn.getPerceptrons().get(3).get(3).output, 0.67822, DELTA);

        final float preResult_hA= nn.getPerceptrons().get(3).get(0).output;
        final float preResult_hB= nn.getPerceptrons().get(3).get(1).output;
        final float preResult_hC= nn.getPerceptrons().get(3).get(2).output;
        final float preResult_hD= nn.getPerceptrons().get(3).get(3).output;
        final float TARGET_oA= 0.9f, TARGET_oB= 0.3f, TARGET_oC= 0.9f, TARGET_oD= 0.3f;
        nn.backpropagate(   sigDx, deltaLearning, lossFunc,
                            nn.getPerceptrons().get(nn.getPerceptrons().size()-1),
                            nn.getEdges(), Arrays.asList(TARGET_oA, TARGET_oB, TARGET_oC, TARGET_oD));

        nn.forward( sig,
                    nn.getEdges(),
                    nn.getPerceptrons(),
                    Arrays.asList(0.5f, 0.7f, 0.9f));

        Assertions.assertTrue(
                Math.abs(TARGET_oA-preResult_hA)>Math.abs(TARGET_oA-nn.getPerceptrons().get(3).get(0).output) &&
                        Math.abs(TARGET_oB-preResult_hB)>Math.abs(TARGET_oB-nn.getPerceptrons().get(3).get(1).output) &&
                        Math.abs(TARGET_oC-preResult_hC)>Math.abs(TARGET_oC-nn.getPerceptrons().get(3).get(2).output) &&
                        Math.abs(TARGET_oD-preResult_hD)>Math.abs(TARGET_oD-nn.getPerceptrons().get(3).get(3).output));
    }

    @Test
    void biasInclusionTest() {
        List<List<Perceptron>> perceptronsPerLayer = new ArrayList<>();
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(1).add(new Perceptron());
        perceptronsPerLayer.get(1).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(2).add(new Perceptron());
        perceptronsPerLayer.get(2).add(new Perceptron());
        perceptronsPerLayer.get(2).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(3).add(new Perceptron());
        perceptronsPerLayer.get(3).add(new Perceptron());

        List<List<Edge>> edges= new ArrayList<>();
        edges.add(new ArrayList<>());
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(0), perceptronsPerLayer.get(1).get(0), 0.1f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(0), perceptronsPerLayer.get(1).get(1), 0.2f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(1), perceptronsPerLayer.get(1).get(0), 0.1f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(1), perceptronsPerLayer.get(1).get(1), 0.2f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(2), perceptronsPerLayer.get(1).get(0), 0.1f));
        edges.get(0).add(new Edge(perceptronsPerLayer.get(0).get(2), perceptronsPerLayer.get(1).get(1), 0.3f));
        edges.add(new ArrayList<>());
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(0), perceptronsPerLayer.get(2).get(0), 0.4f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(0), perceptronsPerLayer.get(2).get(1), 0.5f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(0), perceptronsPerLayer.get(2).get(2), 0.3f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(1), perceptronsPerLayer.get(2).get(0), 0.1f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(1), perceptronsPerLayer.get(2).get(1), 0.2f));
        edges.get(1).add(new Edge(perceptronsPerLayer.get(1).get(1), perceptronsPerLayer.get(2).get(2), 0.6f));
        edges.add(new ArrayList<>());
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(0), perceptronsPerLayer.get(3).get(0), 0.1f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(0), perceptronsPerLayer.get(3).get(1), 0.2f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(1), perceptronsPerLayer.get(3).get(0), 0.3f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(1), perceptronsPerLayer.get(3).get(1), 0.3f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(2), perceptronsPerLayer.get(3).get(0), 0.6f));
        edges.get(2).add(new Edge(perceptronsPerLayer.get(2).get(2), perceptronsPerLayer.get(3).get(1), 0.7f));

        NeuralNet nn= new NeuralNet(edges, perceptronsPerLayer);
        nn.forward( sig,
                    nn.getEdges(),
                    nn.getPerceptrons(),
                    Arrays.asList(0.5f, 0.7f, 0.9f));

        nn.addBias();
        assertEquals(nn.getPerceptrons().get(0).size(), 4);
        assertTrue(nn.getPerceptrons().get(0).get(0).isBias);
        assertEquals(nn.getPerceptrons().get(1).size(), 3);
        assertTrue(nn.getPerceptrons().get(1).get(0).isBias);
        assertEquals(nn.getPerceptrons().get(2).size(), 4);
        assertTrue(nn.getPerceptrons().get(2).get(0).isBias);
        assertEquals(nn.getPerceptrons().get(3).size(), 2);
    }
}