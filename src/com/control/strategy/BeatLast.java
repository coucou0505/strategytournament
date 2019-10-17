package com.control.strategy;

import com.control.PayOff;
import com.tools.Value;

public class BeatLast extends StrategyFindBiggestPayOff {

	public BeatLast(Value payoffval) {
		super(payoffval);
		setName("BeatLast");
	}

	@Override
	public int action(int last) {
		PayOff payOff = (PayOff) payoffVal.getValue();
		
	    if(last == 1) {
	        // Opponent picked A last
	        if(payOff.getAA() > payOff.getBA()) {
	            return 1;
	        } else {
	            return 2;
	        }
	    } else {
	        // Opponent picked B last
	        if(payOff.getAB() > payOff.getBB()) {
	            return 1;
	        } else {
	            return 2;
	        }
	    }
	}

}
