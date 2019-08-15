using System;
using System.Collections.Generic;
using System.Text;
using System.Collections;


namespace libsvm
{
    public class LibSVMKlassifikator
    {
        #region Felder
        double label;
        double[] probs;
        int[] labels;
        #endregion

        #region Properties
        public double Label
        {
            get
            {
                return this.label;
            }
        }

        public int[] Labels
        {
            get
            {
                return this.labels;
            }
        }

        public double[] Wahrscheinlichkeiten
        {
            get
            {
                return this.probs;
            }
        }
        #endregion

        public LibSVMKlassifikator()
        {

        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="text"></param>
        /// <returns></returns>
        public void klassifiziere(string text, Hashtable vektorraum, svm_model model)
        {
            FeatureVector fv = new FeatureVector(text);
            fv.Vektorraum = vektorraum;
            fv.ErzeugeVektor();
            LibSVMWrapper lsvmwrap = new LibSVMWrapper();
            lsvmwrap.SVMModel = model;
            lsvmwrap.klazzifiziere(fv.Vektor);
            label = lsvmwrap.Label;
            labels = lsvmwrap.Labels;
            probs = lsvmwrap.Wahrscheinlichkeiten;
        }
    }
}
