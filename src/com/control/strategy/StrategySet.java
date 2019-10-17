package com.control.strategy;

import com.control.PayOff;
import com.model.Model;
import com.tools.Value;

public class StrategySet {

	private AbstractStrategy[] strategy;
	Model model;
	
	public StrategySet(Model model) {
		this.model = model;
		init();
	}

	public void init(){
		Value payOffValue = (Value) model.getValue("PayOff");
		
		strategy = new AbstractStrategy[8];

		strategy[0] = new Random();
		strategy[1] = new A100();
		strategy[2] = new B100();
		strategy[3] = new Greedy(payOffValue);
		strategy[4] = new Generous(payOffValue);
		strategy[5] = new Minmax(payOffValue);
		strategy[6] = new TitForTat();
		strategy[7] = new BeatLast(payOffValue);
	}
	
	
	public AbstractStrategy getStrategy(int index){		
		return strategy[index];
	}
	
	
	public int getStrategyNumber(){
		return strategy.length;
	}
	
}
