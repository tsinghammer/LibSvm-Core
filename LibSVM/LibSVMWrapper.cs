using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

namespace libsvm
{
    public class LibSVMWrapper
    {
        #region Felder

        private svm_model _model;
        public double[] testvektor;
        private double erg;
        private double[] probs;
        private int[] labels;
        private double[] cs;
        private double[] gs;

        #endregion

        #region Properties
        public svm_model SVMModel
        {
            get
            {
                return this._model;
            }
            set
            {
                this._model = value;
            }
        }

        public double Label
        {
            get
            {
                return this.erg;
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

        #region Constructors

        public LibSVMWrapper() { }

        public LibSVMWrapper(double[] cs, double[] gs)
        {
            this.cs = cs;
            this.gs = gs;
        }

        #endregion

        #region TrainLibSVM

        public void TrainLibSVM(double[][] vektoren, double[] labels, double currentC, double currentG, out int errorCount)
        {
            int nrdocs = vektoren.Length;
            svm_problem prob = new svm_problem();
            prob.l = vektoren.Length - 1;
            prob.y = labels;
            svm_node[][] nodes = new svm_node[nrdocs][];

            for (int i = 0; i < vektoren.Length; i++)
            {
                int dim = vektoren[i].Length;

                nodes[i] = new svm_node[dim + 1];

                for (int j = 0; j < dim; j++)
                {
                    svm_node n = new svm_node();
                    n.index = j;
                    n.value_Renamed = vektoren[i][j];

                    nodes[i][j] = n;
                }
                svm_node ln = new svm_node();
                ln.index = -1;
                ln.value_Renamed = 0;
                nodes[i][dim] = ln;
            }

            prob.x = nodes;

            svm_parameter param = new svm_parameter();
            param.cache_size = 256.0;
            param.C = 1000.0;
            //param.weight = new double[] { 1.0, 1.0, 1.0, 1.0, 1.0 };
            //param.weight_label = new int[] { 1, 1, 1, 1, 1 };
            param.svm_type = svm_parameter.C_SVC;
            param.kernel_type = svm_parameter.SIGMOID;
            param.gamma = 0.00000001;
            param.eps = 0.0001;
            //param.nr_weight = 0;
            param.probability = 1;

            //double[] cs;
            //double[] gs;

            double[] cergs = new double[labels.Length];
            int minfehler = labels.Length;
            int fehler = 0;
            double c = 0.0;
            double g = 0.0;

            #region Parameterabstimmung
            //cs = new double[] { Math.Pow(2.0, -15.0), Math.Pow(2.0, -11.0), Math.Pow(2.0, -9.0), Math.Pow(2.0, -7.0), Math.Pow(2.0, -5.0), Math.Pow(2.0, -3.0), Math.Pow(2.0, -1.0), Math.Pow(2.0, 1.0), Math.Pow(2.0, 3.0), Math.Pow(2.0, 5.0), Math.Pow(2.0, 7.0), Math.Pow(2.0, 12.0), Math.Pow(2.0, 15.0) };
            //gs = new double[] { Math.Pow(2.0, -15.0), Math.Pow(2.0, -12.0), Math.Pow(2.0, -9.0), Math.Pow(2.0, -7.0), Math.Pow(2.0, -5.0), Math.Pow(2.0, -3.0), Math.Pow(2.0, -1.0), Math.Pow(2.0, 1.0), Math.Pow(2.0, 3.0) };
            //cs = new double[] { Math.Pow(2.0, -3.0), Math.Pow(2.0, -1.0), Math.Pow(2.0, 1.0), Math.Pow(2.0, 3.0), Math.Pow(2.0, 5.0), Math.Pow(2.0, 7.0), Math.Pow(2.0, 12.0) };
            //gs = new double[] { Math.Pow(2.0, -7.0), Math.Pow(2.0, -5.0), Math.Pow(2.0, -3.0), Math.Pow(2.0, -1.0), Math.Pow(2.0, 1.0), Math.Pow(2.0, 3.0) };

            //for (int i = 0; i < cs.Length; i++)
            //{
            //    param.C = cs[i];

            //    for (int j = 0; j < gs.Length; j++)
            //    {
            //        fehler = 0;
            //        param.gamma = gs[j];
            //        string res = svm.svm_check_parameter(prob, param);
            //        if (res == null)
            //        {
            //            svm.svm_cross_validation(prob, param, vektoren.Length/4, cergs);

            //            for (int k = 0; k < labels.Length; k++)
            //            {
            //                if (cergs[k] != labels[k])
            //                    fehler++;
            //            }
            //            if (fehler < minfehler)
            //            {
            //                minfehler = fehler;
            //                c = param.C;
            //                g = param.gamma;
            //            }
            //        }
            //    }
            //}

            param.C = currentC;
            fehler = 0;
            param.gamma = currentG;
            string res = svm.svm_check_parameter(prob, param);
            if (res == null)
            {
                svm.svm_cross_validation(prob, param, vektoren.Length / 4, cergs);

                for (int k = 0; k < labels.Length; k++)
                {
                    if (cergs[k] != labels[k])
                        fehler++;
                }
                if (fehler < minfehler)
                {
                    minfehler = fehler;
                    c = param.C;
                    g = param.gamma;
                }
            }

            #endregion

            #region Feinabstimmung
            //cs = new double[] { c * 0.3, c * 0.4, c * 0.5, c * 0.6, c * 0.7, c * 0.8, c * 0.9, c, c * 2.0, c * 3.0, c * 4.0, c * 5.0, c * 6.0 };
            //gs = new double[] { g * 0.5, g * 0.6, g * 0.7, g * 0.8, g * 0.9, g, g * 2.0, g * 3.0, g * 4.0 };
            double[] csF = new double[] { c * 0.6, c * 0.7, c * 0.8, c * 0.9, c, c * 2.0, c * 3.0 };
            double[] gsF = new double[] { g * 0.7, g * 0.8, g * 0.9, g, g * 2.0, g * 3.0 };

            for (int i = 0; i < csF.Length; i++)
            {
                param.C = csF[i];

                for (int j = 0; j < gsF.Length; j++)
                {
                    fehler = 0;
                    param.gamma = gsF[j];
                    res = svm.svm_check_parameter(prob, param);
                    if (res == null)
                    {
                        svm.svm_cross_validation(prob, param, vektoren.Length / 4, cergs);

                        for (int k = 0; k < labels.Length; k++)
                        {
                            if (cergs[k] != labels[k])
                                fehler++;
                        }
                        if (fehler < minfehler)
                        {
                            minfehler = fehler;
                            c = param.C;
                            g = param.gamma;
                        }
                    }
                    //Thread.Sleep(1);
                }
                //Thread.Sleep(10);
            }
            #endregion

            #region Superfeinabstimmung
            //cs = new double[] { c - 7.0, c - 6.0, c - 5.0, c - 4.0, c - 3.0, c - 2.0, c - 1.0, c, c + 1.0, c + 2.0, c + 3.0, c + 4.0, c + 5.0 };
            //gs = new double[] { g - 5.0, g - 4.0, g - 3.0, g - 2.0, g - 1.0, g, g + 1.0, g + 2.0, g + 3.0 };
            /*cs = new double[] { c - 1.0, c - 0.3, c - 0.1, c, c + 0.1, c + 0.3, c + 1.0, };
            gs = new double[] { g - 1.0, g - 0.3, g - 0.1, g, g + 0.1, g + 0.3, g + 1.0 };
            for (int i = 0; i < cs.Length; i++)
            {
                param.C = cs[i];

                for (int j = 0; j < gs.Length; j++)
                {
                    fehler = 0;
                    param.gamma = gs[j];
                    string res = svm.svm_check_parameter(prob, param);
                    if (res == null)
                    {
                        svm.svm_cross_validation(prob, param, 6, cergs);

                        for (int k = 0; k < labels.Length; k++)
                        {
                            if (cergs[k] != labels[k])
                                fehler++;
                        }
                        if (fehler < minfehler)
                        {
                            minfehler = fehler;
                            c = param.C;
                            g = param.gamma;
                        }
                    }
                }
            }*/
            #endregion


            param.C = c;
            param.gamma = g;

            this._model = new svm_model();
            this._model = svm.svm_train(prob, param);

            int anzKlassen = svm.svm_get_nr_class(this._model);
            double[] probs = new double[anzKlassen];

            double erg;
            erg = svm.svm_predict_probability(this._model, nodes[0], probs);
            //erg = svm.svm_predict_probability(this._model, nodes[11], probs);
            //klazzifiziere(this.testvektor);
            //klazzifiziere(vektoren[6]);

            errorCount = minfehler;
        }

        #endregion

        public void klazzifiziere(double[] wortvektor)
        {
            int dim = wortvektor.Length;
            svm_node[] nodes = new svm_node[dim + 1];

            for (int i = 0; i < dim; i++)
            {
                svm_node n = new svm_node();
                n.index = i;
                n.value_Renamed = wortvektor[i];

                nodes[i] = n;
            }
            svm_node ln = new svm_node();
            ln.index = -1;
            ln.value_Renamed = 0;
            nodes[dim] = ln;

            int anzKlassen = svm.svm_get_nr_class(this._model);
            labels = new int[anzKlassen];
            probs = new double[anzKlassen];

            svm.svm_get_labels(this._model, labels);
            erg = svm.svm_predict_probability(this._model, nodes, probs);
        }
    }
}
