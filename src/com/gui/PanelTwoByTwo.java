package com.gui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.GridLayout;
import java.awt.LayoutManager;
import java.awt.Panel;
import javax.swing.JButton;
import java.awt.event.ActionListener;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;
import java.awt.event.ActionEvent;

public class PanelTwoByTwo extends Panel {

	private BattleTextField aa = new BattleTextField();
	private BattleTextField ab = new BattleTextField();
	private BattleTextField ba = new BattleTextField();
	private BattleTextField bb = new BattleTextField();
	
	public PanelTwoByTwo() {
		// TODO Auto-generated constructor stub
		super();
	
		
		this.setLayout(new GridLayout(0,2,0,0));
		
		ab.setTwin(ba);
		ba.setTwin(ab);
		
		add(aa);
		add(ab);
		add(ba);
		add(bb);
		

		
		aa.focusLost();
		ab.focusLost();
		ba.focusLost();
		bb.focusLost();
		
	}

	public Integer getAA(){
		return aa.getValue();		
	}
	
	public Integer getAB(){
		return ab.getValue();		
	}
	
	public Integer getBA(){
		return ba.getValue();		
	}
	
	public Integer getBB(){
		return bb.getValue();		
	}
	
	public PanelTwoByTwo(LayoutManager layout) {
		super(layout);
		// TODO Auto-generated constructor stub
	}

}


