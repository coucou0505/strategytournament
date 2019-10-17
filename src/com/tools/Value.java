package com.tools;

import java.util.Observable;

/**
 * <p>Title: </p>
 * <p>Description: </p>
 * <p>Copyright: Copyright (c) 2004</p>
 * <p>Company: </p>
 * @author not attributable
 * @version 1.0
 */

public class Value
    extends Observable {

  Object oValue;

  public Value(Object oNewValue) {
    oValue = oNewValue;
  }

  /**
   *
   * @param pObject Object
   */
  public void setValue(Object pObject) {

    // copy the new value
    oValue = pObject;

    // indicate that the value have changed
    setChanged();

    // notify the observers
    notifyObservers();
  }

  /**
   * return the value of the object
   * @return Object
   */
  public Object getValue() {
    return oValue;
  }

}
