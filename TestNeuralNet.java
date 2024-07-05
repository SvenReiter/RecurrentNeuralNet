import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TestNeuralNet {
    final double DELTA= 1e-7;

    @Test
    void forwardingTestA() {
        List<List<Perceptron>> perceptronsPerLayer= new ArrayList<>();
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(1).add(new Perceptron());

        List<Edge> edges= new ArrayList<>();
        edges.add(new Edge(perceptronsPerLayer.get(0).get(0), perceptronsPerLayer.get(1).get(0), 0.7f));
        edges.add(new Edge(perceptronsPerLayer.get(0).get(1), perceptronsPerLayer.get(1).get(0), 0.8f));

        NeuralNet nn= new NeuralNet(edges, perceptronsPerLayer);
        nn.forward(x-> (float)(1 / (1 + Math.exp(-x[0]))),
                   nn.getEdges(),
                   nn.getPerceptronsPerLayer(),
                   Arrays.asList(0.5f, 0.6f));

        assertEquals(nn.getPerceptronsPerLayer().get(1).get(0).sum, 0.95224657682, DELTA);
    }

    @Test
    void forwardingTestB() {
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

        List<Edge> edges= new ArrayList<>();
        edges.add(new Edge(perceptronsPerLayer.get(0).get(0), perceptronsPerLayer.get(1).get(0), 0.9f));
        edges.add(new Edge(perceptronsPerLayer.get(0).get(1), perceptronsPerLayer.get(1).get(0), 0.5f));
        edges.add(new Edge(perceptronsPerLayer.get(0).get(2), perceptronsPerLayer.get(1).get(0), 0.7f));
        edges.add(new Edge(perceptronsPerLayer.get(0).get(0), perceptronsPerLayer.get(1).get(1), 0.2f));
        edges.add(new Edge(perceptronsPerLayer.get(0).get(1), perceptronsPerLayer.get(1).get(1), 0.6f));
        edges.add(new Edge(perceptronsPerLayer.get(0).get(2), perceptronsPerLayer.get(1).get(1), 0.8f));

        edges.add(new Edge(perceptronsPerLayer.get(1).get(0), perceptronsPerLayer.get(2).get(0), 0.4f));
        edges.add(new Edge(perceptronsPerLayer.get(1).get(1), perceptronsPerLayer.get(2).get(0), 0.3f));
        edges.add(new Edge(perceptronsPerLayer.get(1).get(0), perceptronsPerLayer.get(2).get(1), 0.4f));
        edges.add(new Edge(perceptronsPerLayer.get(1).get(1), perceptronsPerLayer.get(2).get(1), 0.1f));

        edges.add(new Edge(perceptronsPerLayer.get(2).get(0), perceptronsPerLayer.get(3).get(0), 0.7f));
        edges.add(new Edge(perceptronsPerLayer.get(2).get(1), perceptronsPerLayer.get(3).get(0), 0.2f));

        NeuralNet nn= new NeuralNet(edges, perceptronsPerLayer);
        final Func activationFunc= x-> (float)(1 / (1 + Math.exp(-x[0])));
        nn.forward( activationFunc,
                    nn.getEdges(),
                    nn.getPerceptronsPerLayer(),
                    Arrays.asList(0.2f, 0.3f, 0.4f));

        assertEquals(nn.getPerceptronsPerLayer().get(1).get(0).sum, 1.20115324, DELTA);
        assertEquals(nn.getPerceptronsPerLayer().get(1).get(1).sum, 0.93358246, DELTA);

        assertEquals(nn.getPerceptronsPerLayer().get(2).get(0).sum, 0.52283241, DELTA);
        assertEquals(nn.getPerceptronsPerLayer().get(2).get(1).sum, 0.37927211, DELTA);

        assertEquals(nn.getPerceptronsPerLayer().get(3).get(0).sum, 0.55820636, DELTA);

        assertEquals(nn.getOutput(activationFunc).get(0), 0.6360374, DELTA);

        /*done wieso stimmen meine errechneten Werte nicht überein?
            Weil ich bei der Rechnung aufm Zettel vergessen habe das Gewicht zum Resultat
            der Aktivierungsfunktion zu multiplizieren.
         */
    }
}
