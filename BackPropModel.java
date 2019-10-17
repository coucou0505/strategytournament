package LearningTest.Control.Functions;

import java.io.*;
import java.util.*;

import LearningTest.Tools.RandomNumberGenerator;

/**
 *
 * <p>Title: </p>
 * <p>Description: </p>
 * <p>Copyright: Copyright (c) 2005</p>
 * <p>Company: </p>
 * @author not attributable
 * @version 1.0
 */
public class BackPropModel {


  public static final double MAXIMUM_INPUT = 1;

  public static final double FIRE = 1;
  public static final double NEUTRAL = 0.0;
  public static final double DOWN = -1;

  public static final double INIT = -1.0;
  public static final double DELTA = 0.0;
  public static final double OUTPUT = 1.0;

  protected double instantError; // erreur instantanée
  protected double absoluteError; // erreur absolue
  protected double totalSetError; // erreur absolue

  protected double inpA[]; // activations d'entrées
  protected double hidA[]; // activations cachées
  protected double hidN[]; // somme des produits d'entrée
  protected double hidD[]; // output error
  protected double hidW[][]; // poids de connections de la couche cachée
  protected double outA[]; // activation de sortie
  protected double outN[]; // somme des produits de sortie
  protected double outD[]; // erreur de sortie
  protected double oldD[]; // erreur de sortie anterieure
  protected double outW[][]; // poids de connections de la couche de sortie
  protected double deltaHidW[][]; // variation de poids au cour de l'apprentissage pour les poids de la couche cache
  protected double deltaOutW[][]; // variation de poids au cour de l'apprentissage pour les poids de la couche de sortie

  protected int Ninp; // nombre de neurones d'entrée
  protected int Nhid; // nombre de neurones cachés
  protected int Nout; // nombre de neurones de sortie

  protected double eta; // taux d'apprentisage
  protected double elast; // elasticit\uFFFDde la sigmoide
  protected double alpha; // moment a utiliser

  RandomNumberGenerator randGene = new RandomNumberGenerator();


  public BackPropModel() {
  }

  /**
   * initialisation du reseau
   *
   * @param ninp int
   * @param nhid int
   * @param nout int
   */
  public void init() {
    int i, m;
    //int iIndexRandomNumber = ( (Integer) model.getValue("RandomNumberVariation").
     //                         getValue()).intValue();

    //eta = ( (Double) model.getValue("LearningRate"+sExtention).getValue()).doubleValue();
    //elast = ( (Double) model.getValue("Elasticity"+sExtention).getValue()).doubleValue();
    //alpha = ( (Double) model.getValue("Momentum"+sExtention).getValue()).doubleValue();

    /**
     * number of input of the neural network
     */
    //Ninp = getInputFromModel("Hf")+getInputFromModel("Lf");

   // Ninp = ( (Integer) model.getValue("NumberOfInput"+sExtention).getValue()).intValue();
   // Nout = ( (Integer) model.getValue("NumberOfOutput"+sExtention).getValue()).intValue();
   // Nhid = ( (Integer) model.getValue("NumberOfHidden"+sExtention).getValue()).intValue();

    this.inpA = new double[Ninp + 1];
    this.hidW = new double[Nhid][Ninp + 1];
    this.hidA = new double[Nhid + 1];
    this.hidN = new double[Nhid];
    this.hidD = new double[Nhid];
    this.outW = new double[Nout][Nhid + 1];
    this.outA = new double[Nout];
    this.outN = new double[Nout];
    this.outD = new double[Nout];
    this.oldD = new double[Nout];
    this.deltaHidW = new double[Nhid][Ninp + 1];
    this.deltaOutW = new double[Nout][Nhid + 1];

    // init random generator
    randGene.reset();



    for (i = 0; i < Ninp; i++) {
      inpA[i] = 2*randGene.random()-1;//frandom( -1.0, 1.0);
    }
    inpA[Ninp] = 1;

    for (i = 0; i < Nhid; i++) {
      hidA[i] = 2*randGene.random()-1;//frandom( -1.0, 1.0);

      //System.out.print(hidA[i]+" ");

      for (m = 0; m < Ninp+1; m++) {
        hidW[i][m] = 2*randGene.random()-1;//frandom( -1.0, 1.0);
        //System.out.print(hidW[i][m]+" ");
      }
      //System.out.println();

    }
    hidA[Nhid] = 1;

    for (i = 0; i < Nout; i++) {
      for (m = 0; m < Nhid+1; m++) {
        outW[i][m] = 2*randGene.random()-1;//frandom( -1.0, 1.0);

      }
    }


  }

  /**
   *
   * @param input Vector
   * @param output Vector
   * @return double
   */
  public double learnSet(Vector input, Vector output) {

    Vector tempin = new Vector();
    Vector tempout = new Vector();

    // normalise the input
    for (int i = 0; i < input.size(); i++) {
      tempin.add(new Double( ( (Double) input.elementAt(i)).
                            doubleValue() / MAXIMUM_INPUT));
    }

    for (int i = 0; i < output.size(); i++) {
      tempout.add( (Double) output.elementAt(i));
    }

    // apprend le in pour le out
    learn(tempin, tempout);

    // calculate the new error

    // return good value
    return absoluteError;
  }

  /**
   *
   * @param <any> input
   * @return double
   */
  protected double getMaxAbs(Vector input) {
    double dMax = Math.abs( ( (Double) input.elementAt(0)).doubleValue());

    for (int i = 1; i < input.size(); i++) {
      if (dMax < Math.abs( ( (Double) input.elementAt(i)).doubleValue())) {
        dMax = Math.abs( ( (Double) input.elementAt(i)).doubleValue());
      }
    }

    return dMax;
  }

  /**
   *
   * @return Vector
   */

  public Vector getResult() {
    Vector v = new Vector();
    for (int i = 0; i < this.Nout; i++) {
      v.add(new Double(this.outA[i]));
    }
    return v;
  }

  /**
   *  fonction sigmoidale
   */
  protected double sigmoid(double x) {
    return 1.0 / (1.0 + Math.exp( -1.0 * elast * x));
  }

  /**
   *  fonction derivée de la sigmoide
   */
  protected double d1sigmoid(double x) {
    return elast * sigmoid(x) * (1 - sigmoid(x));
  }

// fonction aleatoire
  public double frandom(double mini, double maxi) {
    return Math.random() * (maxi - mini) + mini;
  }

  /**
   * propagation des activations des entrees.
   */
  public void feedForward() {
    int i, j;
    double sum2;

    inpA[Ninp] = 1;
    for (i = 0; i < Nhid; i++) {
      sum2 = 0.0;
      for (j = 0; j < Ninp + 1; j++) {
        sum2 += hidW[i][j] * inpA[j];
      }
      hidN[i] = sum2;
      hidA[i] = sigmoid(sum2);
    }

    hidA[Nhid] = 1;
    for (i = 0; i < Nout; i++) {
      sum2 = 0.0;
      for (j = 0; j < Nhid + 1; j++) {
        sum2 += outW[i][j] * hidA[j];
      }
      outN[i] = sum2;

    }

  }

  /**
   * calcul du delta
   * @param m int : indice a calculer
   * @return double retuourne l'erreur
   */
  public double computeError() {

    // define the array of error
    Vector vError = new Vector();

    // compute the error
    for (int i = 0; i < outN.length; i++) {
      vError.add(new Double(
          Math.abs(outA[i] - sigmoid(outN[i]))));
    }
    /*   if( getMaxAbs(vError) > 0.2){
         System.out.println(getMaxAbs(vError));
       }*/

    //  return the maximum error
    return getMaxAbs(vError);
  }

  /**
   * mise a jour des poids
   *
   */
  public void updateWeights() {

    double[] deltaErrorOut = new double[Nout];
    double[] deltaErrorHid = new double[Nhid];
    double sum;
    double prod1;
    double prod2;

    //////////////////////////////////////////////
    // update weight for hidden to output layer
    //////////////////////////////////////////////

    // compute the delta for output
    for (int m = 0; m < Nout; m++) {

      sum = 0;
      for (int k = 0; k < Nhid + 1; k++) {
        sum += hidA[k] * outW[m][k];
      }
      deltaErrorOut[m] = d1sigmoid(sum) * (outA[m] - sigmoid(outN[m]));

    }


    // compute the deltaOutW
    for (int m = 0; m < Nout; m++) {
      for (int k = 0; k < Nhid + 1; k++) {
        deltaOutW[m][k] = alpha * deltaOutW[m][k] +
            eta * hidA[k] * deltaErrorOut[m];
      }
    }

    // update output weight
    for (int m = 0; m < Nout; m++) {
      for (int k = 0; k < Nhid + 1; k++) {
        outW[m][k] += deltaOutW[m][k];
      }
    }

    //////////////////////////////////////////////
    // update weight for input to hidden layer
    //////////////////////////////////////////////

    // delta for the hidden layer
    for (int k = 0; k < Nhid; k++) {

      prod1 = 0;
      for (int m = 0; m < Nout; m++) {
        prod1 += outW[m][k] * deltaErrorOut[m];
      }

      prod2 = 0;
      for (int j = 0; j < Ninp + 1; j++) {
        prod2 += hidW[k][j] * inpA[j];
      }

      deltaErrorHid[k] = d1sigmoid(prod2) * prod1;
    }

    for (int j = 0; j < Ninp + 1; j++) {
      for (int k = 0; k < Nhid; k++) {
        deltaHidW[k][j] = alpha * deltaHidW[k][j] +
            eta * inpA[j] * deltaErrorHid[k];
      }
    }

    for (int j = 0; j < Ninp + 1; j++) {
      for (int k = 0; k < Nhid; k++) {
        hidW[k][j] = hidW[k][j] + deltaHidW[k][j];
      }
    }

  }

  /**
   * propagate the value.
   *
   * @param in Vector : input vector
   * @throws ArrayIndexOutOfBoundsException
   * @return Vector : output vector
   */

  public Vector propagate(Vector in) throws ArrayIndexOutOfBoundsException {

    double sum2;
    double[] TempH = new double[this.Nhid + 1];
    double[] TempO = new double[this.Nout];
    Vector sortie = new Vector();

    for (int i = 0; i < Ninp; i++) {

      inpA[i] = ( (Double) in.elementAt(i)).doubleValue() / MAXIMUM_INPUT;
    }
    inpA[Ninp] = 1;

    // propagate all the data
    // from input to hidden layer
    for (int i = 0; i < Nhid; i++) {
      sum2 = 0.0;
      for (int j = 0; j < Ninp + 1; j++) {
        sum2 += hidW[i][j] * inpA[j];
      }
      TempH[i] = sigmoid(sum2);
    }
    TempH[Nhid] = 1;

    // propagate from hidden to output layer
    for (int i = 0; i < Nout; i++) {
      sum2 = 0.0;
      for (int j = 0; j < Nhid + 1; j++) {
        sum2 += outW[i][j] * TempH[j];
      }
      TempO[i] = sigmoid(sum2);
    }

    for (int i = 0; i < Nout; i++) {
      outA[i] = TempO[i];
      sortie.add(new Double(outA[i]));
    }
    return sortie;
  }

  /**
   *  apprentissage du in pour le out
   *
   * @param values Vector : entree
   * @param out Vector    : sortie
   * @throws ArrayIndexOutOfBoundsException
   */
  public void learn(Vector values, Vector out) throws
      ArrayIndexOutOfBoundsException {

    for (int i = 0; i < values.size(); i++) {
      inpA[i] = ( (Double) values.elementAt(i)).doubleValue();
    }

    inpA[Ninp] = 1;

    for (int i = 0; i < Nout; i++) {
      outA[i] = ( (Double) out.elementAt(i)).doubleValue();
    }

    // propagation of the input
    feedForward();

    // update the weights
    updateWeights();

    // propagate one more time to calculate the error
    feedForward();

    // compute the error for this in - out
    absoluteError = computeError();

  }

  /**
   * set and get for protected variable
   *
   */

  public int getNinput(){
    return Ninp;
  }

  public int getNhidden(){
    return Nhid;
  }

  public int getNoutput(){
    return Nout;
  }

  public double getEta() {
    return eta;
  }

  public double getElast() {
    return this.elast;
  }

  public double getNhid() {
    return this.Nhid;
  }

  public double getMoment() {
    return alpha;
  }

  public double getInstantError() {
    return instantError;
  }

  public double getAbsoluteError() {
    return absoluteError;
  }

  public int getHiddenLength() {
    return Nhid;
  }

  public int getInputLength() {
    return Ninp;
  }

  public int getOutputLength() {
    return Nout;
  }

  public double getTotalSetError() {
    return this.totalSetError;
  }

  public double[] getOutput() {
    return outA;
  }

  public double[][] gethidW() {
    return hidW;
  }

  public double[][] getoutW() {
    return outW;
  }

  public void setEida(double eta) {
    this.eta = eta;
  }

  public void setElast(double elast) {
    this.elast = elast;
  }

  public void setNhih(int Nhid) {
    if (this.hidN == null) {
      this.Nhid = Nhid;
    }
  }

  public void setMomentum(double moment) {
    this.alpha = moment;
  }

  public void setHiddenWeight(double[][] dWeight) {
    hidW = dWeight;
  }

  public void setOutputWeight(double[][] dWeight) {
    outW = dWeight;
  }

  public void setNinput(int iNbInput){
    Ninp = iNbInput;
  }
  public void setNhidden(int iNbHidden){
    Nhid = iNbHidden;
  }
  public void setNoutput(int iNbOutput){
    Nout = iNbOutput;
  }

  /**
  *
  * @param i int
  * @return double
  */
  private double controledRandomNumber(int i){

  double x = (1741)+i;

  for (int j = 0; j < 16; j++) {
    x *= Math.PI;

    if (x > 16) {
      x = x - Math.floor(x);
    }
  }
  x = x - Math.floor(x);

  return x;

  }

  private double controledRandomNumber(int i,int k){
  double x = (1741)+i+k;

  for (int j = 0; j < 16; j++) {
    x *= Math.PI;

    if (x > 16) {
      x = x - Math.floor(x);
    }
  }
  x = x - Math.floor(x);

  return x;

  }


}
