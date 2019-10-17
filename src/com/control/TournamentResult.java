package com.control;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Hashtable;

import com.control.strategy.StrategySet;

public class TournamentResult {

	ArrayList<ResultatStrategy> listResultat = new ArrayList<ResultatStrategy>();
	Hashtable<String, ResultatStrategy> hTableResult = new Hashtable<String, ResultatStrategy>();
	StrategySet set = null;
	

	public TournamentResult(StrategySet set) {
		this.set = set;
		init();
	}

	public void init(){
		listResultat.clear();
		hTableResult.clear();
		ResultatStrategy res;

		for (int i = 0; i < set.getStrategyNumber(); i++) {
			res = new ResultatStrategy(set.getStrategy(i));
			listResultat.add(res);
			hTableResult.put(res.getName(), res);
		}
	}


	public ResultatStrategy getResult(String name){
		return hTableResult.get(name);
	}
	
	public void sort(){
		Collections.sort(listResultat);
	}
	
	
	public ArrayList<ResultatStrategy> getListResultat(){
		return listResultat;
		
	}
	
}
