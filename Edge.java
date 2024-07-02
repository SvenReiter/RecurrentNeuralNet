public class Edge {
    Perceptron from, to;
    float weight;

    public Edge(Perceptron from, Perceptron to) {
        this.from= from;
        this.to= to;
        weight= (float)(Math.random()*2-1);
    }
}
