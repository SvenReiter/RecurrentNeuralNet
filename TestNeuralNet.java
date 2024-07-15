import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TestNeuralNet {
    final double DELTA= 1e-7;
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
                nn.getPerceptronsPerLayer(),
                Arrays.asList(0.5f, 0.6f));

        assertEquals(nn.getPerceptronsPerLayer().get(1).get(0).sum, 0.83, DELTA);
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
                nn.getPerceptronsPerLayer(),
                Arrays.asList(0.2f, 0.3f, 0.4f));

        assertEquals(nn.getPerceptronsPerLayer().get(1).get(0).sum, 0.61, DELTA);
        assertEquals(nn.getPerceptronsPerLayer().get(1).get(1).sum, 0.54, DELTA);

        assertEquals(nn.getPerceptronsPerLayer().get(2).get(0).sum, 0.44872004, DELTA);
        assertEquals(nn.getPerceptronsPerLayer().get(2).get(1).sum, 0.32235756, DELTA);

        assertEquals(nn.getPerceptronsPerLayer().get(3).get(0).sum, 0.54321417, DELTA);

        assertEquals(nn.getPerceptronsPerLayer().get(3).get(0).output, 0.6325598, DELTA);
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
                nn.getPerceptronsPerLayer(),
                Arrays.asList(1f, 0.7f, 0.6f));

        final double DELTA= 1e-3;
        assertEquals(nn.getPerceptronsPerLayer().get(2).get(0).output, 0.655, DELTA);
        assertEquals(nn.getPerceptronsPerLayer().get(2).get(1).output, 0.55, DELTA);

        nn.backpropagate(sigDx, deltaLearning, lossFunc,
                nn.getPerceptronsPerLayer().get(nn.getPerceptronsPerLayer().size()-1),
                nn.getEdges(), Arrays.asList(0.9f, 0.2f));

        assertEquals(nn.getPerceptronsPerLayer().get(2).get(0).delta, 0.0554667897522, DELTA);
        assertEquals(nn.getPerceptronsPerLayer().get(2).get(1).delta, -0.08662214130163193, DELTA);

        assertEquals(nn.getEdges().get(1).get(1).to.delta, 0.055, DELTA);
        assertEquals(nn.getEdges().get(1).get(5).to.delta, -0.087, DELTA);

        int LAYER_ID= nn.getEdges().size()-1;

        assertEquals(edges.get(LAYER_ID).get(0).weight, 0.255, DELTA);
        assertEquals(edges.get(LAYER_ID).get(1).weight, 0.442, DELTA);
        assertEquals(edges.get(LAYER_ID).get(2).weight, 0.325, DELTA);
        assertEquals(edges.get(LAYER_ID).get(3).weight, 0.013, DELTA);
        assertEquals(edges.get(LAYER_ID).get(4).weight, -0.466, DELTA);
        assertEquals(edges.get(LAYER_ID).get(5).weight, 0.861, DELTA);

        assertEquals(nn.getEdges().get(1).get(1).from.delta, 0.010367614, DELTA);
        assertEquals(nn.getEdges().get(1).get(2).from.delta, -0.015399089, DELTA);

        LAYER_ID--;
        assertEquals(edges.get(LAYER_ID).get(0).weight, 0.3+0.010367614, DELTA);
        assertEquals(edges.get(LAYER_ID).get(1).weight, 0.8+0.0072573298, DELTA);
        assertEquals(edges.get(LAYER_ID).get(2).weight, 0.5+0.006220569, DELTA);
        assertEquals(edges.get(LAYER_ID).get(3).weight, -0.2-0.015399089, DELTA);
        assertEquals(edges.get(LAYER_ID).get(4).weight, -0.6-0.010779362, DELTA);
        assertEquals(edges.get(LAYER_ID).get(5).weight, 0.7-0.009239454, DELTA);
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
        final float INPUT_A= 0.7f, INPUT_B= 0.6f;
        nn.forward( sig,
                nn.getEdges(),
                nn.getPerceptronsPerLayer(),
                Arrays.asList(1f, INPUT_A, INPUT_B));

        final float preResult_hA= nn.getPerceptronsPerLayer().get(2).get(0).output;
        final float preResult_hB= nn.getPerceptronsPerLayer().get(2).get(1).output;
        final float TARGET_oA= 0.9f, TARGET_oB= 0.2f;
        nn.backpropagate(sigDx, deltaLearning, lossFunc,
                nn.getPerceptronsPerLayer().get(nn.getPerceptronsPerLayer().size()-1),
                nn.getEdges(), Arrays.asList(TARGET_oA, TARGET_oB));

        nn.forward( sig,
                nn.getEdges(),
                nn.getPerceptronsPerLayer(),
                Arrays.asList(1f, INPUT_A, INPUT_B));

        // tests whether trained output is closer to target than previous output
        Assertions.assertTrue(
    Math.abs(TARGET_oA-preResult_hA)>Math.abs(TARGET_oA-nn.getPerceptronsPerLayer().get(2).get(0).output) &&
            Math.abs(TARGET_oB-preResult_hB)>Math.abs(TARGET_oB-nn.getPerceptronsPerLayer().get(2).get(1).output));
    }
}