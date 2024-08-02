import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TestNeuralNet {
    final double DELTA= 1e-5;
    final float ETA= 1f;                                                                                                //Schrittweite/Lernrate
    final Func sig = x-> (float)(1 / (1 + Math.exp(-x[0])));                                                            //Aktivierungsfunktion, berechnet die Ausgabe(den Aktivierungslevel) des Perzeptron.
    final Func sigDx = x-> (sig.calc(x)* (1-sig.calc(x)));                                                              //abgeleitete Aktivierungsfunktion
    final Func lossFunc= x-> (x[0]-x[1]);                                                                               //Kostenfunktion/Verlustfunktion    x[0]:=SOLL x[1]:=IST
    final Func deltaLearning= x-> (ETA* x[0]* x[1]);                                                                    //Delta-Lernregel                   x[0]:=aktivierungslevel(soll) x[1]:=aktivierungslevel(ist)

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

        assertEquals(nn.getPerceptrons().get(1).get(0).input, 0.83, DELTA);
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

        assertEquals(nn.getPerceptrons().get(1).get(0).input, 0.61, DELTA);
        assertEquals(nn.getPerceptrons().get(1).get(1).input, 0.54, DELTA);

        assertEquals(nn.getPerceptrons().get(2).get(0).input, 0.44872004, DELTA);
        assertEquals(nn.getPerceptrons().get(2).get(1).input, 0.32235756, DELTA);
        assertEquals(nn.getPerceptrons().get(3).get(0).input, 0.54321417, DELTA);
        assertEquals(nn.getPerceptrons().get(3).get(0).output, 0.6325598, DELTA);
    }

    @Test
    void forwardBackpropagateTestA() {
        List<List<Perceptron>> perceptronsPerLayer = new ArrayList<>();
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).get(0).input = 1f;
        perceptronsPerLayer.get(0).get(0).output= 1f;
        perceptronsPerLayer.get(0).get(0).isBias= true;
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(1).add(new Perceptron());
        perceptronsPerLayer.get(1).get(0).input = 1f;
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

        assertEquals(nn.getPerceptrons().get(1).get(1).output, 0.7613327, DELTA);                                 //sig(1*0.3 +0.7*0.8+ 0.6*0.5)
        assertEquals(nn.getPerceptrons().get(1).get(2).output, 0.450166, DELTA);                                  //sig(1*(-0.2)+ 0.7* (-0.6)+ 0.6*0.7)
        assertEquals(nn.getPerceptrons().get(2).get(0).output, 0.6546592, DELTA);                                 //sig(1*0.2+ 0.7613327*0.4+ 0.450166*0.3)
        assertEquals(nn.getPerceptrons().get(2).get(1).output, 0.5499865, DELTA);                                 //sig(1*0.1+ 0.7613327*(-0.4)+ 0.450166*0.9)

        nn.backpropagate(   sigDx, deltaLearning, lossFunc,
                            nn.getPerceptrons().get(nn.getPerceptrons().size()-1),
                            nn.getEdges(), Arrays.asList(0.9f, 0.2f));
                                                                                                                        //Outputlayer:= f'(netzinput bzw. <sum>)* (aktivierungslevel(soll)- aktivierungslevel(ist) bzw. <output>(ist))
        assertEquals(nn.getEdges().get(1).get(0).to.delta, 0.0554667897522, DELTA);                               // = f'(0.63958293)* (0.9- 0.65465915)
        assertEquals(nn.getEdges().get(1).get(1).to.delta, 0.0554667897522, DELTA);                               // = f'(0.63958293)* (0.9- 0.65465915)
        assertEquals(nn.getEdges().get(1).get(2).to.delta, 0.0554667897522, DELTA);                               // = f'(0.63958293)* (0.9- 0.65465915)
        assertEquals(nn.getEdges().get(1).get(3).to.delta, -0.08662214106, DELTA);                                // = f'(0.2006163)* (0.2- 0.54998654)
        assertEquals(nn.getEdges().get(1).get(4).to.delta, -0.08662214106, DELTA);                                // = f'(0.2006163)* (0.2- 0.54998654)
        assertEquals(nn.getEdges().get(1).get(5).to.delta, -0.08662214106, DELTA);                                // = f'(0.2006163)* (0.2- 0.54998654)

        assertEquals(nn.getPerceptrons().get(2).get(0).delta, 0.0554667897522, DELTA);                            // = f'(0.63958293)* (0.9- 0.65465915)
        assertEquals(nn.getPerceptrons().get(2).get(1).delta, -0.0866221413016, DELTA);                           // = f'(0.2006163)* (0.2- 0.54998654)
                                                                                                                        //Delta-Lernregel   ->  deltaLearning: weight+= (<ETA>(1) * <to.delta> * <from.output>)
        assertEquals(edges.get(1).get(0).weight, 0.2554667897522, DELTA);                                         //0.2+= (1)*0.0554667897522*1(bias)
        assertEquals(edges.get(1).get(1).weight, 0.44222286838, DELTA);                                           //0.4+= (1)*0.0554667897522*0.76133275
        assertEquals(edges.get(1).get(2).weight, 0.32496926409, DELTA);                                           //0.3+= (1)*0.05546679*0.45016602
        assertEquals(edges.get(1).get(3).weight, 0.01337786, DELTA);                                              //0.1+= (1)*(-0.08662214)*1(bias)
        assertEquals(edges.get(1).get(4).weight, -0.46594827205, DELTA);                                          //-0.4+= (1)*(-0.08662214)*0.76133275
        assertEquals(edges.get(1).get(5).weight, 0.86100565599, DELTA);                                           //0.9+= (1)*(-0.08662214)*0.45016602

                                                                                                                        //Hidden-layer:= f'(from.netzinput)* SUMME(<to.delta>* <weight>)ÜBER ALLE <to>'S
        assertEquals(nn.getEdges().get(1).get(1).from.delta, 0.01179093201, DELTA);
                                                                                                                        //f'(1.1600001)*(0.05546679*0.44222867+ (-0.08662214)*(-0.46594828))
                                                                                                                        //0.18170521991*(0.05546679*0.44222867+ (-0.08662214)*(-0.46594828))
        assertEquals(nn.getEdges().get(1).get(2).from.delta, -0.0139988309, DELTA);
                                                                                                                        //f'(-0.2)*(0.05546679*0.32496926+ (-0.08662214)*0.8610056)
                                                                                                                        //0.24751657244*(0.05546679*0.32496926+ (-0.08662214)*0.8610056)
                                                                                                                        //<weightPRE>+= deltaLearning(to.delta, from.output)
        assertEquals(edges.get(0).get(0).weight, 0.3+0.01179093, DELTA);                                          //  0.3      += ( (1)*0.01179093*1(bias) )
        assertEquals(edges.get(0).get(1).weight, 0.8+0.008253651 , DELTA);                                        //  0.8      += ( (1)*0.01179093*0.7 )
        assertEquals(edges.get(0).get(2).weight, 0.5+0.007074558, DELTA);                                         //  0.5      += ( (1)*0.01179093*0.6 )
        assertEquals(edges.get(0).get(3).weight, -0.2-0.013998833, DELTA);                                        //  -0.2     += ( (1)*-0.013998833*1(bias) )
        assertEquals(edges.get(0).get(4).weight, -0.6-0.0097991831, DELTA);                                       //  -0.6     += ( (1)*-0.013998833*0.7 )
        assertEquals(edges.get(0).get(5).weight, 0.7-0.0083992998, DELTA);                                        //  0.7      += ( (1)*-0.013998833*0.6 )
    }

    @Test
    void forwardBackpropagateTestB() {
        List<List<Perceptron>> perceptronsPerLayer = new ArrayList<>();
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).get(0).input = 1f;
        perceptronsPerLayer.get(0).get(0).output= 1f;
        perceptronsPerLayer.get(0).get(0).isBias= true;
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.get(0).add(new Perceptron());
        perceptronsPerLayer.add(new ArrayList<>());
        perceptronsPerLayer.get(1).add(new Perceptron());
        perceptronsPerLayer.get(1).get(0).input = 1f;
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

        float preResult_hA= nn.getPerceptrons().get(2).get(0).output;
        float preResult_hB= nn.getPerceptrons().get(2).get(1).output;
        float TARGET_oA= 0.9f, TARGET_oB= 0.2f;
        nn.backpropagate(   sigDx, deltaLearning, lossFunc,
                            nn.getPerceptrons().get(nn.getPerceptrons().size()-1),
                            nn.getEdges(), Arrays.asList(TARGET_oA, TARGET_oB));

        nn.forward( sig,
                    nn.getEdges(),
                    nn.getPerceptrons(),
                    Arrays.asList(1f, 0.7f, 0.6f));

        Assertions.assertTrue(                                                                                          // tests whether trained output is closer to target than previous output
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

        assertEquals(nn.getPerceptrons().get(3).get(0).output, 0.6492844, DELTA);                                 // sig(0.5704*0.1+ 0.59895*0.3+ 0.63195*0.6)
        assertEquals(nn.getPerceptrons().get(3).get(1).output, 0.676149, DELTA);                                  // sig(0.5704*0.2+ 0.59895*0.3+ 0.63195*0.7)
        assertEquals(nn.getPerceptrons().get(3).get(2).output, 0.6465778, DELTA);                                 // sig(0.5704*0.4+ 0.59895*0.1+ 0.63195*0.5)
        assertEquals(nn.getPerceptrons().get(3).get(3).output, 0.6782159, DELTA);                                 // sig(0.5704*0.1+ 0.59895*0.2+ 0.63195*0.9)

        float preResult_hA= nn.getPerceptrons().get(3).get(0).output;
        float preResult_hB= nn.getPerceptrons().get(3).get(1).output;
        float preResult_hC= nn.getPerceptrons().get(3).get(2).output;
        float preResult_hD= nn.getPerceptrons().get(3).get(3).output;
        float TARGET_oA= 0.9f, TARGET_oB= 0.3f, TARGET_oC= 0.9f, TARGET_oD= 0.3f;
        nn.backpropagate(   sigDx, deltaLearning, lossFunc,
                            nn.getPerceptrons().get(nn.getPerceptrons().size()-1),
                            nn.getEdges(), Arrays.asList(TARGET_oA, TARGET_oB, TARGET_oC, TARGET_oD));
                                                                                                                        //Outputlayer:= f'(from.netzinput)* (aktivierungslevel(soll)- aktivierungslevel(ist))
        assertEquals(nn.getEdges().get(2).get(0).to.delta, 0.05709149423, DELTA);                                 //f'(0.615895)*(0.9-0.6492844)= 0.22771416791*(0.9-0.6492844)
        assertEquals(nn.getEdges().get(2).get(1).to.delta, -0.0823658355, DELTA);                                 //f'(0.7361253)*(0.3-0.676148)= 0.21897188209*(0.3-0.676148)
        assertEquals(nn.getEdges().get(2).get(2).to.delta, 0.057910917, DELTA);                                   //f'(0.60402095)*(0.9-0.6465778)= 0.22851556417*(0.9-0.6465778)
        assertEquals(nn.getEdges().get(2).get(3).to.delta, -0.08254149496, DELTA);                                //f'(0.745585)*(0.3-0.6782159)= 0.21823909298*(0.3-0.6782159)
                                                                                                                        //Delta-Lernregel   ->  deltaLearning: weight+= (<ETA>(1) * <to.delta> * <from.output>)
        assertEquals(nn.getEdges().get(2).get(0).weight, 0.13256397242, DELTA);                                   //0.1+= (1)*0.057091665*0.5703805
        assertEquals(nn.getEdges().get(2).get(1).weight, 0.15302013356, DELTA);                                   //0.2+= (1)*(-0.0823658355)*0.5703805
        assertEquals(nn.getEdges().get(2).get(2).weight, 0.43303125779, DELTA);                                   //0.4+= (1)*0.057910917*0.5703805
        assertEquals(nn.getEdges().get(2).get(3).weight, 0.05291994083, DELTA);                                   //0.1+= (1)*(-0.08254149496)*0.5703805
        assertEquals(nn.getEdges().get(2).get(4).weight, 0.33419537246 , DELTA);                                  //0.3+= (1)*0.057091665*0.5989556
        assertEquals(nn.getEdges().get(2).get(5).weight, 0.25066652157 , DELTA);                                  //0.3+= (1)*(-0.0823658355)*0.5989556
        assertEquals(nn.getEdges().get(2).get(6).weight, 0.13468606803, DELTA);                                   //0.1+= (1)*0.057910917*0.5989556
        assertEquals(nn.getEdges().get(2).get(7).weight, 0.15056130936, DELTA);                                   //0.2+= (1)*(-0.08254149496)*0.5989556
        assertEquals(nn.getEdges().get(2).get(8).weight, 0.63607907769, DELTA);                                   //0.6+= (1)*0.057091665*0.63195
        assertEquals(nn.getEdges().get(2).get(9).weight, 0.64794891025, DELTA);                                   //0.7+= (1)*(-0.0823658355)*0.63195
        assertEquals(nn.getEdges().get(2).get(10).weight, 0.53659680399, DELTA);                                  //0.5+= (1)*0.057910917*0.63195
        assertEquals(nn.getEdges().get(2).get(11).weight, 0.84783790226, DELTA);                                  //0.9+= (1)*(-0.08254149496)*0.63195
                                                                                                                        //Hidden-layer:= f'(<from.netzinput>)* sum (<to.delta>* <weight>) of every <to> from <nn.getEdges().get(LAYER_ID)> that points to <from>
        assertEquals(nn.getEdges().get(2).get(0).from.delta, 0.00384086943, DELTA);                               //f'(0.2834038)* (0.057091665*0.13256398+ (-0.08236583)*0.15302014+ 0.057911396*0.43303153+ (-0.08254141)*0.052919988) = 0.24504659* (0.01567403748)
        assertEquals(nn.getEdges().get(2).get(4).from.delta, -0.00148799881, DELTA);                              //f'(0.40111524)* (0.057091665*0.33419537246+ (-0.08236583)*0.25066652157+ 0.057911396*0.13468606803+ (-0.08254141)*0.15056130936) = 0.24022180424* (-0.00619427039)
        assertEquals(nn.getEdges().get(2).get(8).from.delta, -0.01301592204, DELTA);                              //f'(0,5405763)* (0.057091665*(0.6+0.036078874)+ (-0.08236583)*(0.7-0.052050795)+ 0.057911396*(0.5+0.0365969)+ (-0.08254141)*(0.9-0.052161753)) = 0.23259012113* (-0.05596076087)
        assertEquals(nn.getEdges().get(1).get(0).from.delta, -0.0007441269387, DELTA);                            //f'(0.21000001)* (0.003840869*0.40212134+ (-0.0014879096)*0.4991782+ (-0.013015922)*0.29281121) = (-0.9997441269387)* (-0.00300944451)
        assertEquals(nn.getEdges().get(1).get(3).from.delta, -0.001783163761, DELTA);                             //f'(0.51)* (0.003840869*0.1023998+ (-0.0014879096)*0.19907035+ (-0.013015922)*0.59186757) = 0.23442333755* (-0.00760659659)
        assertEquals(nn.getEdges().get(0).get(0).from.delta, -0.0001008585956, DELTA);                            //f'(0.5)* ((-0.00074412697)*0.099627934+ (-0.0017831636)* 0.19910842) = 0.23500371984* (-0.0004291787197)
        assertEquals(nn.getEdges().get(0).get(2).from.delta, -0.00009498889593, DELTA);                           //f'(0.7)* ((-0.00074412697)*0.09947911+ (-0.0017831636)*0.19875179) = 0.22171286393* 0.0004284320461
        assertEquals(nn.getEdges().get(0).get(4).from.delta, -0.00012453354, DELTA);                              //f'(0.9)* ((-0.00074412697)*0.09933028+ (-0.0017831636)*0.29839516) = 0.20550030845*(-0.00012683926)
        nn.forward( sig,
                    nn.getEdges(),
                    nn.getPerceptrons(),
                    Arrays.asList(0.5f, 0.7f, 0.9f));

        Assertions.assertTrue(                                                                                          // tests whether trained output is closer to target than previous output
              Math.abs(TARGET_oA-preResult_hA) > Math.abs(TARGET_oA-nn.getPerceptrons().get(3).get(0).output) &&
              Math.abs(TARGET_oB-preResult_hB) > Math.abs(TARGET_oB-nn.getPerceptrons().get(3).get(1).output) &&
              Math.abs(TARGET_oC-preResult_hC) > Math.abs(TARGET_oC-nn.getPerceptrons().get(3).get(2).output) &&
              Math.abs(TARGET_oD-preResult_hD) > Math.abs(TARGET_oD-nn.getPerceptrons().get(3).get(3).output)
        );
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

        assertEquals(nn.getPerceptrons().get(1).get(0).output, 0.5523079, DELTA);                                 //sig(0.5*0.1+ 0.7*0.1+ 0.9*0.1)
        assertEquals(nn.getPerceptrons().get(1).get(1).output, 0.6248065, DELTA);                                 //sig(0.5*0.2+ 0.7*0.2+ 0.9*0.3)
        assertEquals(nn.getPerceptrons().get(2).get(0).output, 0.5703805, DELTA);                                 //sig(0.5523079*0.4+ 0.6248065*0.1)
        assertEquals(nn.getPerceptrons().get(2).get(1).output, 0.5989556, DELTA);                                 //sig(0.5523079*0.5+ 0.6248065*0.2)
        assertEquals(nn.getPerceptrons().get(2).get(2).output, 0.6319465, DELTA);                                 //sig(0.5523079*0.3+ 0.6248065*0.6)
        assertEquals(nn.getPerceptrons().get(3).get(0).output, 0.6492838, DELTA);                                 //sig(0.5703805*0.1+ 0.5989556*0.3+ 0.6319465*0.6)
        assertEquals(nn.getPerceptrons().get(3).get(1).output, 0.676148, DELTA);                                  //sig(0.5703805*0.2+ 0.5989556*0.3+ 0.6319465*0.7)

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