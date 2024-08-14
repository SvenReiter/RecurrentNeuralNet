public class Perceptron {
    boolean isBias;
    float input;
    float output;   //for inputPerceptrons value of "output" equals value of "input"
    float delta;    /*First for calculate hidden-layers <delta>s sum over their referenced
                     * outgoing-layers-perceptrons.outputs multiplied with referenced edges-weights.
                     *Second for perceptrons delta-value.
                     */
}