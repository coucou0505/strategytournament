package com.control;

public class PayOff {


	private final static int A_INDEX = 0; 
	private final static int B_INDEX = 1;

	private int[][] payOff = null;

	public PayOff() {
		init();
	}

	public void init(){
		payOff = new int[2][2];

		for (int i = 0; i < payOff.length; i++) {
			for (int j = 0; j < payOff[i].length; j++) {
				payOff[i][j] = 1;
			}
		}
	}


	public void setAA(int val) {
		if(val>0 && val <=10)
			payOff[A_INDEX][A_INDEX] = val;
	}

	public int getAA() {
		return payOff[A_INDEX][A_INDEX];
	}


	public void setBA(int val) {
		if(val>0 && val <=10)
			payOff[B_INDEX][A_INDEX] = val;
	}

	public int getBA() {
		return payOff[B_INDEX][A_INDEX];
	}

	public void setAB(int val) {
		if(val>0 && val <=10)
			payOff[A_INDEX][B_INDEX] = val;
	}

	public int getAB() {
		return payOff[A_INDEX][B_INDEX];
	}

	public void setBB(int val) {
		if(val>0 && val <=10)
			payOff[B_INDEX][B_INDEX] = val;
	}

	public int getBB() {
		return payOff[B_INDEX][B_INDEX];
	}



	public void upAA() {
		if(payOff[A_INDEX][A_INDEX] < 10) {
			payOff[A_INDEX][A_INDEX] += 1;

		}
	}

	public void downAA() {
		if(payOff[A_INDEX][A_INDEX] > 1) {
			payOff[A_INDEX][A_INDEX] -= 1;

		}
	}

	public void  upAB() {
		if(payOff[A_INDEX][B_INDEX] < 10) {
			payOff[A_INDEX][B_INDEX] += 1;

		}
	}

	public void  downAB() {
		if(payOff[A_INDEX][B_INDEX] > 1) {
			payOff[A_INDEX][B_INDEX] -= 1;

		}
	}

	public void  upBA() {
		if(payOff[B_INDEX][A_INDEX] < 10) {
			payOff[B_INDEX][A_INDEX] += 1;

		}
	}

	public void  downBA() {
		if(payOff[B_INDEX][A_INDEX] > 1) {
			payOff[B_INDEX][A_INDEX] -= 1;

		}
	}

	public void  upBB() {
		if(payOff[B_INDEX][B_INDEX] < 10) {
			payOff[B_INDEX][B_INDEX] += 1;

		}
	}

	public void  downBB() {
		if(payOff[B_INDEX][B_INDEX] > 1) {
			payOff[B_INDEX][B_INDEX] -= 1;

		}
	}

	public int[][] getPayOff() {
		return payOff;
	}
	
}
