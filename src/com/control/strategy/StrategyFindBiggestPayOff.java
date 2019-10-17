package com.control.strategy;

import com.control.PayOff;
import com.tools.Value;

public abstract class StrategyFindBiggestPayOff extends AbstractStrategy {

	Value payoffVal;
	
	public StrategyFindBiggestPayOff(Value payoffval) {
		super();
		this.payoffVal = payoffval;
	}

	
	protected int findBiggestPayoff() {
		PayOff payOff = (PayOff) payoffVal.getValue();

		int aa = payOff.getAA();
		int ab = payOff.getAB();
		int ba = payOff.getBA();
		int bb = payOff.getBB();

		if(aa >= ab && aa >= ba && aa >= bb) {
			return 1;
		} else if(ab >= aa && ab >= ba && ab >= bb) {
			return 2;
		} else if(ba >= aa && ba >= ab && ba >= bb) {
			return 3;
		} else {
			return 4;
		}
	}	
	
	
}
