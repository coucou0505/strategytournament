package com.control.strategy;

public class Random extends AbstractStrategy {

	public Random() {
		// TODO Auto-generated constructor stub
		setName("Random");
	}

	@Override
	public int action(int last) {
	    double r = Math.random();
	    if(r < .5) {
	        return 1;
	    } else {
	        return 2;
	    }
	}

}
