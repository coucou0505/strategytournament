package com.gui;

import java.awt.BorderLayout;
import java.awt.Panel;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JButton;
import javax.swing.JFrame;

import com.control.PayOff;
import com.control.Tournament;
import com.model.Model;

public class Gui extends JFrame implements Runnable{
	Model model;
	PanelTwoByTwo twobytwo;
	PanelTextDisplay panelText = null;
	Tournament tournament;
	Thread t = new Thread(this);

	public Gui(Model model) {
		addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent arg0) {
				System.exit(0);
			}
		});

		tournament = new Tournament(model); 

		this.model =model;
		setSize(150,280);

		init();		

		setVisible(true);

	}

	private void init(){

		getContentPane().setLayout(new BorderLayout(0, 0));

		panelText = new PanelTextDisplay();		
		getContentPane().add(panelText, BorderLayout.CENTER);


		Panel paneltwobytwo= new Panel();
		twobytwo = new PanelTwoByTwo();

		paneltwobytwo.setLayout(new BorderLayout());
		paneltwobytwo.add(twobytwo, BorderLayout.CENTER);


		//JButton btnNewButtonCheck = new JButton("Tournament");
		//btnNewButtonCheck.addKeyListener(new PanelCheckKeyListener(this));

		//btnNewButtonCheck.addActionListener(new PanelTwoByTwoActionListener(this));

		//paneltwobytwo.add(btnNewButtonCheck, BorderLayout.SOUTH);

		getContentPane().add(paneltwobytwo, BorderLayout.NORTH);
		buttonAction();

	}


	public void buttonAction(){

		t.start();


	}

	public void run(){
		PayOff payoff = ((PayOff) model.getValue("PayOff").getValue()); 


		while(true){
			payoff.setAA(twobytwo.getAA());
			payoff.setAB(twobytwo.getAB());
			payoff.setBA(twobytwo.getBA());
			payoff.setBB(twobytwo.getBB());

			tournament.clearResults();
			
			for (int i = 0; i < 50; i++) {
				tournament.runTournament();
				
				try {
					Thread.sleep(10);
				} catch (Exception e) {
					// TODO: handle exception
				}
				
				
			}
			
			panelText.setText(tournament.getResultString());
		}

	}

}




class PanelTwoByTwoActionListener implements ActionListener{
	Gui bft;

	public PanelTwoByTwoActionListener(Gui bft){
		this.bft = bft;
	}

	public void actionPerformed(ActionEvent arg0) {
		bft.buttonAction();
	}	

}




class PanelCheckKeyListener extends KeyAdapter{
	Gui gui ; 
	public PanelCheckKeyListener(Gui gui){
		this.gui = gui;
	}

	public void keyPressed(KeyEvent arg0) {
		if(arg0.getKeyChar() == KeyEvent.VK_ENTER){
			gui.buttonAction();
		}
	}


}