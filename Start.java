
public class Start {
    public static void main(String[] args) {
        /*
         *  lineare Regression:= Spezialfall der Regressionsanalyse, mit dem versucht wird eine beobachtete
         *                       abhängige Variable durch eine oder mehrere unabhängige Variablen zu erklären.
         *                       Bei der linearen Regression wird dabei ein lineares Modell angenommen.
         *  Aktivierungsfunktion:= Sie macht aus einem linearen Wert einen nicht linearen Wert
         */
        NeuralNet nn= new NeuralNet(
                x-> (1 / (1 + Math.exp(-x[0]))),
                x-> (Math.pow((x[0]-x[1]), 2)),
                4,3,3,3
        );
        //todo <Weights> Implementierung so dass geschickt iteriert/verrechnet werden kann
    }
}