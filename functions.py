import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
import re


from matplotlib.ticker import FuncFormatter
from optbinning  import BinningProcess
from scipy       import stats as st
from scipy.stats import chi2_contingency
from scipy.stats import ks_2samp
from scipy.stats import kstwo, ksone

LINE       = "--"
ROTATION   = 30
FONT_SIZE  = 9
WIDTH      = 8*1.3
HEIGHT     = 2*1.3
COLOR      = "#001820"
COLOR_BAR  = "#429e53"
COLOR_LINE = "white"
COLOR_DOT  = "#001820"


def sep(s, thou=",", dec="."):
    integer, decimal = s.split(".")
    integer = re.sub(r"\B(?=(?:\d{3})+$)", thou, integer)
    return integer + dec + decimal


def create_table_categorical(var, df, order_cat = None):

    eda = df[var].value_counts().reset_index()
    eda.columns = ["Class", "Abs. Freq."]

    if order_cat != None:
        eda["sort"] = [
            np.where([x == y for x in order_cat])[0][0] for y in eda["Class"].tolist()
            ]
        eda = (
            eda
            .sort_values(by = "sort")
            .reset_index(drop = True)
            .drop("sort", axis = 1)
            )
    else:
        eda.sort_values(by = "Class", ascending = True, inplace = True)
        eda.reset_index(drop = True, inplace = True)


    eda["Acc. Abs. Freq."] = eda["Abs. Freq."].cumsum()
    eda["Rel. Freq."] = eda["Abs. Freq."]/eda["Abs. Freq."].sum()
    eda["Acc. Rel. Freq."] = eda["Rel. Freq."].cumsum()

    eda[["Abs. Freq.", "Acc. Abs. Freq."]] = (
        eda[["Abs. Freq.", "Acc. Abs. Freq."]].applymap(lambda x: "{:.0f}".format(np.round(x, 0)))
        ) 
    
    eda[["Rel. Freq.", "Acc. Rel. Freq."]] = (
        eda[["Rel. Freq.", "Acc. Rel. Freq."]].applymap(lambda x: "{:.2f}".format(np.round(x, 2)))
        ) 
    
    eda = eda.astype(str)
    blankIndex = [''] * len(eda)
    eda.index  = blankIndex
    
    df_styled = (
        eda.style
            # Cor do header e index
            .set_table_styles([{
                'selector': 'th:not(.index_name)',
                'props': f'background-color: {COLOR}; color: white;'
            }])
        )
      
    return df_styled 


def create_graph_categorical(var, df, order_cat = None):
    
    data = (
            df[[var]]
            .value_counts()
            .reset_index()
            )
    
    data.columns = ["idx",var]
    data.iloc[:, 1:] = data.iloc[:, 1:].astype(np.int64)
    
    if order_cat != None:
        data["sort"] = [
            np.where([x == y for x in order_cat])[0][0] for y in data["idx"].tolist()
            ]
        data = (
            data
            .sort_values(by = "sort")
            .reset_index(drop = True)
            .drop("sort", axis = 1)
            )
    else:
        data.sort_values(by = "idx", ascending = True, inplace = True)
        data.reset_index(drop = True, inplace = True)
   
    x = data["idx"].astype(str)
    y = data[var]
    
    plt.rc('legend',fontsize = FONT_SIZE)
    plt.rc('font', size = FONT_SIZE)
    
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), facecolor="#93a7b8")
    bars = ax.bar(x, y, color = COLOR_BAR)
    
    ax.set_title(f'Distribuição de Frequências - {var}')
    ax.yaxis.set_visible(False)
    
    data = y
    v_max = data[~np.isinf(data)].max()
    v_min = data[~np.isinf(data)].min()
  
    ax.set_ylim(
        v_min - np.abs(v_min*0.3), 
        v_max + np.abs(v_max*0.2)
    )
    
    
    for bars in ax.containers:
        ax.bar_label(bars, labels=[f'{x:.1%}' for x in bars.datavalues/np.sum(bars.datavalues)])
    
    ax.set_facecolor("#93a7b8")
    plt.xticks(x)
    plt.show()


def create_table_numeric_continuous(var, df):
    
    cols = ['SUM', 'CNT', 'AVG', 'STDEV','PERC_zeros', 'PERC_negatives',
            'MIN', 'P1', 'P25', 'P50', 'P75', 
            'P90', 'P95', 'P99', 'MAX']
    
    eda = exploratory_data_analysis_numerical(
        df  = df, 
        var = var, 
        q   = [1, 25, 50, 75, 90, 95, 99]
    )
    
    eda[['CNT']] = eda[['CNT']].applymap(lambda x: "{:.0f}".format(np.round(x, 0)))
    
    eda[['SUM', 'AVG', 'STDEV','PERC_zeros', 
         'PERC_negatives','MIN', 'P1', 'P25', 
         'P50', 'P75', 'P90', 'P95', 'P99', 'MAX']] = eda[[
             'SUM', 'AVG', 'STDEV','PERC_zeros', 
             'PERC_negatives','MIN', 'P1', 'P25', 
             'P50', 'P75', 'P90', 'P95', 'P99', 'MAX']].applymap(
                 lambda x: sep("{:.2f}".format(np.round(x, 2))))
    
    
    blankIndex = [''] * len(eda)
    eda.index  = blankIndex
    eda        = eda[cols]
    
    df_styled = (
        eda.style
            # Cor do header e index
            .set_table_styles([{
                'selector': 'th:not(.index_name)',
                'props': f'background-color: {COLOR}; color: white;'
            }])
        )
            
    return df_styled 


def create_graph_numeric_continuous(var, df):
       
    data = df[[var]]
    
    plt.rc('legend',fontsize = FONT_SIZE)
    plt.rc('font', size = FONT_SIZE)
    
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), facecolor="#93a7b8")
    ax.hist(data, color = COLOR_BAR)
    
    ax.set_title(f'Distribuição de Frequências - {var}')
    ax.yaxis.set_visible(False)
    
    ax.set_facecolor("#93a7b8")
    plt.show()
    

def perc_zeros(df):
    return   df.loc[df.values == 0, ].count()/df.shape[0]

def perc_negatives(df):
    return   df.loc[df.values < 0, ].count()/df.shape[0]

def trim_mean(df, total_cut = 0.1):
    return st.trim_mean(df.values, proportiontocut = total_cut/2)

def positive_mean(df):
    return df.loc[df.values > 0, ].mean()

def exploratory_data_analysis_numerical(df, var, q = [1, 5, 25, 50, 75, 90, 95, 98, 99]):
    '''
    This fucntion creates a exploratory_data_analysis from a numerical variable
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame which must contains the var.
    var : str
        The variable name.
    q : list, optional
        Percentils. The default is [1, 5, 25, 50, 75, 90, 95, 98, 99].
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with the statistics about the var.
    
    '''
    
    # Delete NaN
    df.dropna(inplace = True)
    df.reset_index(drop = True, inplace = True)
    
    # Aggregate by SUM, CNT, AVG, STDEV, MIN and MAX and AVG_pos
    df_calc = df.loc[:, var].agg(SUM            = 'sum', 
                                 CNT            = 'count',
                                 AVG            = np.mean,
                                 STDEV          = np.std, 
                                 MIN            = np.min, 
                                 MAX            = np.max,
                                 PERC_zeros     = perc_zeros,
                                 PERC_negatives = perc_negatives).reset_index().T

    # Renamne
    df_calc.rename(columns = df_calc.iloc[0, :], inplace = True)
    df_calc.drop(index = 'index', axis = 0, inplace = True)
    
    
    # Calculate percentis
    percentis = np.array(np.percentile(a = df.loc[:, var], q = q)).reshape(1,-1)
    
    P = ['P'+ str(i) for i in q]
    
    df_percentis = pd.DataFrame(
        columns  = P,
        data     = percentis,
        index    = [var])
    
    # Merge
    df_final = pd.concat([df_calc, df_percentis], axis = 1)
    
    # Order
    df_final = df_final[df_calc.columns.tolist() + 
                        df_percentis.columns.tolist()]    
    
    
    return df_final.astype('float64')#.round(2)


def normalize_str(df, col_name):
    """
    Preprocesses the data of a given column. Keep only numbers.
    This function removes leading zeros as well.
    """
    
    # Convert to string
    df[col_name] = df[col_name].astype('str')
        
    # Keep only numbers
    df[col_name] = df[col_name].apply(lambda x : ''.join(re.findall(r'\d+', re.sub('\.0$', '', x))))

    # Remove leading zeros
    df[col_name] = df[col_name].apply(lambda x : x.lstrip('0'))
    
    # Replace   
    c1 = df[col_name].apply(len) <= 1
    df.loc[c1, col_name] = np.nan
    
    # Replance 'nan' with NaN
    c2 = df[col_name] == 'nan'
    df.loc[c2, col_name] = np.nan
    
    return df


def binning(df, target_variable, categorical_variables=None):
    """
    This function performs a binning optimization for the variables contained 
    in the given dataframe (except for a target variable)

    See http://gnpalencia.org/optbinning/binning_process.html                        

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with all the variables that will be binned, including the 
        target variable.
    target_variable : str
        The name of the target variable.
    categorical_variables : list
        A list of all numeric variables that should be considered categorical.
    Returns
    -------
    binning_process.summary() : pd.DataFrame
        A dataframe with a summary of each binned variable, for example:
        dtype, status, n_bis etc
    df_res : pd.DataFrame
        A dataframe with all variables binned.

    """

    # All variables in df
    variable_names = np.ndarray.tolist(df.columns.values)

    # Remove target_variable
    variable_names = [
        item for item in variable_names if item != target_variable]

    min_n_bins = 2
    max_n_bins = 10
    min_bin_size = 0.05

    if "qty_restricted_approved_add_vol_tickets_t3" in df.columns:
        print(variable_names)
        print("qty_restricted_approved_add_vol_tickets_t3")
        min_bin_size = 0.02

    X = df[variable_names].values
    y = df[target_variable].values

    if categorical_variables != None:
        binning_process = BinningProcess(variable_names,
                                         categorical_variables=categorical_variables,
                                         max_n_bins=max_n_bins,
                                         min_bin_size=min_bin_size,
                                         min_n_bins=min_n_bins)
    else:
        binning_process = BinningProcess(variable_names,
                                         max_n_bins=max_n_bins,
                                         min_bin_size=min_bin_size,
                                         min_n_bins=min_n_bins)

    binning_process.fit(X, y)
    X_transform = binning_process.transform(X, metric="bins")

    df_res = pd.DataFrame(data=X_transform, columns=variable_names)
    df_res[target_variable] = y

    # binning_process.information()

    return (binning_process.summary(), df_res)


def calculate_CramersV(var_name_x, var_name_y, df):
    """
    This function calculates the degree of association between two categorical 
    variables contained in the dataframe. The metric used is Cramer's V.

    See 
        FÁVERO, Luiz Paulo Lopes e BELFIORE, Patrícia Prado. Manual de análise 
        de dados: estatística e modelagem multivariada com excel, SPSS e stata.
        Rio de Janeiro: Elsevier.
    Parameters
    ----------
    var_name_x : str
        The X variable name.
    var_name_y : str
        The Y variable name..
    df : pd.DataFrame
        A dataframe with X and Y variables.
    Returns
    -------
    V : float
        Cramer's V coefficient of the relationship between X and Y.
    """

    X = df[var_name_x]
    Y = df[var_name_y]
    df_cross_tab = pd.crosstab(index=X, columns=Y, margins=True)

    # 2) Calculate Chi^2
    result = chi2_contingency(observed=df_cross_tab.iloc[:-1, :-1])
    x2 = result[0]

    # 3) Calculate Cramer's V
    q = np.min(df_cross_tab.iloc[:-1, :-1].shape)
    n = df_cross_tab.loc["All", "All"]
    V = np.sqrt(x2/(n*(q-1))) if x2 != 0 else 0

    return V


def calculate_CramersV2(var_name_x, var_name_y, df):

    X = var_name_x
    Y = var_name_y

    data = df[[X, Y]]

    # Freq. observada:
    obs_abs = pd.crosstab(data[X], data[Y], margins=True)
    obs = pd.crosstab(data[X], data[Y], normalize='index', margins=True)

    # Freq. esperada: Supondo que não haja associação entre X e Y:
    exp = np.matmul(
        obs_abs.loc[:, "All"].drop("All").values.reshape(-1, 1),
        obs.loc["All"].values.reshape(1, -1)
    )

    obs_abs = obs_abs.drop("All", axis=0).drop("All", axis=1)
    X2 = (((obs_abs - exp)**2)/exp).sum().sum()  # chi quadrado
    r = obs_abs.shape[0]                       # num linhas
    c = obs_abs.shape[1]                       # num colunas
    q = np.min([c, r])                          # grau de liberadade
    n = obs_abs.sum().sum()                    # num registros
    V = np.sqrt(X2/(n*(q-1))) if X2 != 0 else 0

    return V


def bivariate(df, numerical_variables, categorical_variables, target_variable):

    l1 = 0.05
    l2 = 0.20

    # Filter
    df = df[numerical_variables + categorical_variables + [target_variable]]

    # Forced Tipying
    df[numerical_variables] = df[numerical_variables].astype(float)
    df[categorical_variables] = df[categorical_variables].astype(str)

    # Features
    cols = df.drop([target_variable], axis=1).columns

    # Result
    df_res = pd.DataFrame()
    df_original = df.copy()

    # For each variable in df (except target_variable)
    for var_name in cols:

        # 1 - Binning:-------------------------------------------------------------
        df_aux = df_original[[target_variable, var_name]].copy()
        df = df_aux.copy()

        categorical_var = [
            var_name] if var_name in categorical_variables else None
        result = binning(df_aux, target_variable, categorical_var)
        df = result[1]

        # 2 - Total, Event and Non-event ------------------------------------------
        total = df.groupby([var_name])[target_variable].count()
        event = df.groupby([var_name])[target_variable].sum()
        non_event = total - event

        df_res1 = pd.concat([total, event, non_event], axis=1).reset_index()
        df_res1.columns = ['Categoria', 'Total', 'Evento', 'Nao_evento']
        df_res1["Feature"] = var_name

        # 3 - Cramers'V -----------------------------------------------------------
        var_x = var_name
        var_y = target_variable

        V = calculate_CramersV(var_name_x=var_x,
                               var_name_y=var_y,
                               df=df)

        df_res1["Cramer's V"] = V
        df_res = pd.concat([df_res, df_res1])
        # --------------------------------------------------------------------------

    # Create metrics 2
    df_res['% Resposta'] = (df_res['Evento'] / (df_res['Total']))*100
    qtd_growers = df_res.groupby(['Feature']).agg({'Total': 'sum'})
    df_res['% Categoria'] = (df_res['Total'] / qtd_growers.iloc[0, 0])*100

    # Classifying
    df_res['Discriminância'] = df_res["Cramer's V"].apply(
        lambda x: "Baixa" if x <= l1 else ("Média" if x <= l2 else "Alta"))

    # Create metrics 3
    df_res['% Media da base'] = (
        sum(df_res['Evento']) / sum(df_res['Total']))*100

    # Sort
    df_res = df_res[['Feature', 'Categoria',	'Nao_evento', 'Evento',	'Total',
                     '% Media da base', '% Resposta',	'% Categoria',
                     "Cramer's V",	'Discriminância']]

    df_res.reset_index(drop=True, inplace=True)
    # df_res.to_excel("bivariate.xlsx", index = False)

    return df_res


def binning_to_model(df, numerical_variables, categorical_variables, target_variable):

    # Filter
    df = df[numerical_variables + categorical_variables + [target_variable]]

    # Forced Tipying
    df[numerical_variables] = df[numerical_variables].astype(float)
    df[categorical_variables] = df[categorical_variables].astype(str)

    # Features
    cols = df.drop([target_variable], axis=1).columns

    # Result
    df_res = pd.DataFrame()
    df_original = df.copy()

    # For each variable in df (except target_variable)
    for var_name in cols:

        df_aux = df_original[[target_variable, var_name]].copy()
        df = df_aux.copy()

        categorical_var = [
            var_name] if var_name in categorical_variables else None
        result = binning(df_aux, target_variable, categorical_var)
        df = result[1]
        
        if df_res.shape[0] == 0:
            df_res = pd.concat([df_res, df])
        else:
            df_res = pd.merge(df_res, 
                              df[var_name], 
                              how = "left", 
                              left_index=True, 
                              right_index=True, 
                              validate="one_to_one")

    return df_res

def create_table_bivariate_summary(df, cols_float = None):


    blank_index = [''] * len(df)
    df.index = blank_index

    if cols_float != None:
        df[cols_float] = df[cols_float].applymap(
            lambda x: "{:.2f}".format(np.round(x, 2))).copy()

    df_styled = (
        df.style
        # Cor do header e index
        .set_table_styles([{
            'selector': 'th:not(.index_name)',
            'props': f'background-color: {COLOR}; color: white;'
        }])
    )

    return df_styled


def create_table_bivariate_html(df, var):

    df = df.loc[df["Feature"] == var].copy()

    cols = ['Categoria', 'Nao_evento', 'Evento', 'Total',
            '% Resposta', '% Categoria']

    # cols = ['Categoria', 'Nao_evento', 'Evento', 'Total',
    #        '% Resposta', '% Categoria', "Cramer's V", "Discriminância"]

    cols_float = ['% Resposta', '% Categoria']

    blank_index = [''] * len(df)
    df.index = blank_index

    df[cols_float] = df[cols_float].applymap(
        lambda x: "{:.2f}".format(np.round(x, 2))).copy()

    df = df[cols]

    df.rename(columns={"Nao_evento": "Não Churn",
                       "Evento": "Churn",
                       "% Resposta": "% Churn",
                       }, inplace=True)

    df_styled = (
        df.style
        # Cor do header e index
        .set_table_styles([{
            'selector': 'th:not(.index_name)',
            'props': f'background-color: {COLOR}; color: white;'
        }])
    )
    return df_styled


def create_graph_bivariate_html(df, var):

    df = df.loc[df["Feature"] == var]

    mean = df["% Media da base"].mean()

    fig, ax1 = plt.subplots(figsize=(WIDTH, HEIGHT), facecolor="#93a7b8")

    # Barra no primeiro eixo y (à esquerda)
    ax1.bar(df['Categoria'], df['% Resposta'],
            color=COLOR_BAR, alpha=0.7, label='% Churn')

    ax1.axhline(y=mean, color=COLOR_LINE, linestyle='--',
                label='% Média de Churn')

    for bars in ax1.containers:
        # f'{x:.4}%'
        ax1.bar_label(
            bars, labels=[f'{round(x, 2)}%' for x in bars.datavalues])

    ax1.scatter(df['Categoria'], df['% Categoria'],
                color=COLOR_DOT, marker='o', label='% Categoria')

    # Fixar os limites dos eixos y
    ax1.set_ylim(0, 100)

    plt.title(f"{var} - Porcentagem de Churn por Categoria")

    # Adicionar o símbolo de porcentagem
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}%'))

    ax1.legend(loc='upper right')
    plt.legend(facecolor = "#becad4") 

    ax1.set_facecolor("#93a7b8")
    plt.rc('legend',fontsize = FONT_SIZE)
    plt.rc('font', size = FONT_SIZE)
    plt.xticks(rotation=ROTATION)
    plt.show()


def create_graph_h_bivariate_html(df, var):

    df   = df.loc[df["Feature"] == var]
    mean = df["% Media da base"].mean()


    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), facecolor="#93a7b8")
    ax.barh(df['Categoria'], df['% Resposta'],
            color=COLOR_BAR, alpha=0.7, label='% Churn')

    ax.axvline(x=mean, color=COLOR_LINE, linestyle='--',
               label='% Média de Churn')

    for bars in ax.containers:
        # f'{x:.4}%'
        ax.bar_label(bars, labels=[f'{round(x, 2)}%' for x in bars.datavalues])

    ax.scatter(df['% Categoria'], df['Categoria'],
               color=COLOR_DOT, marker='o', label='% Categoria')

    # Fixar os limites dos eixos y
    ax.set_xlim(0, 100)

    plt.title(f"{var} - Porcentagem de Churn por Categoria")

    # Adicionar o símbolo de porcentagem
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}%'))

    ax.legend(loc='upper right')
    plt.legend(facecolor = "#becad4") 

    ax.set_facecolor("#93a7b8")
    plt.rc('legend',fontsize = FONT_SIZE)
    plt.rc('font', size = FONT_SIZE)
    plt.show()
    
    
def diff_row_by_row(df, var):
    return np.insert(np.diff(df[var], axis=0), 0, df.loc[0, var])


def ecdf(data):
    """
    Calculates the empirical cumulative distribution function (ECDF)
    
    Parameters
    ----------
    data : like array
        Data sample.

    Returns
    -------
    pd.DataFrame
        Data sample with the ECDF.
    """

    data = np.array(data)
    p = [(data <= x).sum() / len(data) for x in np.unique(np.sort(data))] 
    x = np.unique(np.sort(data)).tolist()  
    
    return (pd.DataFrame({'x': x, 'P(X <= x)': p}))


def plot_ecdf(cdf_0, cdf_1, name_ecdf0, name_ecdf1):
    """
    Creates and plots the empirical cumulative distribution function (ECDF)
    
    Parameters
    ----------
    ecdf1 : like array
        ecdf1 sample.
    """
    
    # Criando uma figura e um eixo (axis)
    fig, ax = plt.subplots(facecolor="#93a7b8")
    
    # Plotando o gráfico de dispersão no eixo
    x = cdf_0['x']
    y = cdf_0['P(X <= x)']
    ax.step(x, y, color=COLOR, label=f'{name_ecdf0}', where = "pre")
    
    # Plotando o gráfico de dispersão no eixo
    x = cdf_1['x']
    y = cdf_1['P(X <= x)']
    ax.step(x, y, color=COLOR_BAR, label=f'{name_ecdf1}', where = "pre")
    
    # ks
    ks = np.abs(cdf_0["P(X <= x)"] - cdf_1["P(X <= x)"]).max()
    c  = np.abs(cdf_0["P(X <= x)"] - cdf_1["P(X <= x)"]) == ks
    x0, y0 = cdf_0.loc[c,].values[0]
    x1, y1 = cdf_1.loc[c,].values[0]
    y0, y1 = (np.min([y0, y1]), np.max([y0, y1]))
    ax.plot([x0, x1], [y0, y1], color = "black")
    ax.text(x0, (y0 + y1)/2, f'{"{:.2f}".format(round(ks, 2))}', 
            color='black')
    
    # Adicionando rótulos e título
    ax.set_xlabel('x')
    ax.set_ylabel('P(X <= x)')
    ax.set_title('Funções de Distribuição Acumulada (CDF)')
    
    # Adicionando uma legenda
    ax.legend(loc='upper right')
    plt.legend(facecolor = "#becad4") 

    ax.set_facecolor("#93a7b8")
    plt.rc('legend',fontsize = FONT_SIZE)
    plt.rc('font', size = FONT_SIZE)
    
    # Exibindo o gráfico
    plt.show()


def ks_test(X0, X1, alpha = 0.05):

    cdf_0  = ecdf(X0)
    cdf_1  = ecdf(X1)
    
    # Graficos
    plot_ecdf(cdf_0, cdf_1, name_ecdf0="0's", name_ecdf1 ="1's")
    
    # Calculos
    D      = np.abs((cdf_0["P(X <= x)"] - cdf_1["P(X <= x)"]))
    KS     = D.max()
    n      = D.shape[0]
    loc    = D.mean()
    scale  = D.std()
    
    kstwo.cdf(KS, n, loc, scale)                   # P(D <= KS) 
    p_value = 2*kstwo.sf(KS, n, loc, scale)        # 2*P(D >  KS) 
    rc_0    = kstwo.ppf(alpha/2, n, loc, scale)    # d1 | P(D <= d1) = alpha/2
    rc_1    = kstwo.ppf(1- alpha/2, n, loc, scale) # d2 | P(D <= d2) = alpha/2
    
    # Se KS está dentro da região crítica a probabilidade de ocorrer um valor
    # mais extremo é grande (esse é p-value)
    if (rc_0 <= KS) and (KS <= rc_1):
        out = f"""
            H0 = As duas amostras possuem distribuições de frequências iguais.
            H1 = As duas amostras possuem distribuições de frequências diferentes.     
                  
            Aceitamos H0, uma vez que a probabilidade de ocorrer um valor mais extremo 
            que KS é alta, ou seja
              
              p_value = {p_value} > {alpha}
              
            Assim, a probabilidade de rejeitar H0, sendo H0 verdadeira, é maior que o
            o nível de significância definido (alpha = {alpha})!"""
              
    # Se KS está fora da região crítica a probabilidade de ocorrer um valor
    # mais extremo é pequena (esse é p-value)
    else:
        out = f"""
        H0 = As duas amostras possuem distribuições de frequências iguais.
        H1 = As duas amostras possuem distribuições de frequências diferentes.
        
        Rejeitamos H0, uma vez que a probabilidade de ocorrer um valor mais extremo 
        que KS é baixa, ou seja, é raro de ocorrer.
        
          p_value = {p_value} < {alpha}
        
        Assim, a probabilidade de rejeitar H0, sendo H0 verdadeira, é menor que o
        nível de significância definido (alpha = {alpha})!"""
              
              
    return KS, p_value 


def create_ks_table_for_logistic_regression(clf, X, y):
    
    # Probabilidade de ser 1
    y_prob = clf.predict_proba(X)[:, 1]
    
    # armazenar a resposta
    nos = pd.DataFrame()
    
    # probs
    v = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 
         0.65, 0.70, 0.75, 0.80, 0.85, 0.9, 0.95]
    probs = [pd.DataFrame(y_prob).quantile(p).values[0] for p in v]
    probs = list(set(probs))
    probs.sort(reverse = True)
    
    for p in probs:
        
        # Cut
        y_range = y[y_prob >= p]
                
        # 0 (real)
        zeros = (y_range == 0).sum()
        
        # 1 (real)
        ones = (y_range == 1).sum()
        
        # armazenando
        df_aux = pd.DataFrame({
            'Prob'      : p,
            'Evento acumulado'    : ones,
            'Nao-evento acumulado': zeros 
            }, index = [0])
        
        nos = pd.concat([nos, df_aux])
        nos.reset_index(drop = True, inplace = True)
        
 
    # Obtendo valor não acumulado
    nos["Evento"]     =  diff_row_by_row(nos, "Evento acumulado")
    nos["Nao-evento"] =  diff_row_by_row(nos, "Nao-evento acumulado")

    # Total
    nos['Total']           = nos['Nao-evento'] + nos['Evento']
    nos['Total acumulado'] = nos['Nao-evento acumulado'] + nos['Evento acumulado']

    # Removendo faixas que não acrescentam
    c = nos["Prob"].isin(
        nos.sort_values(by = "Total acumulado", ascending = False).groupby("Total acumulado")["Prob"].max()
        )
    nos = nos.loc[c,].sort_values(by = "Prob", ascending = False).reset_index(drop = True)
    
    # % Total Acumulado
    nos['% Total acumulado'] = nos['Total acumulado'] / nos['Total acumulado'].max()
    
    # % Resp. 1 acumulado
    nos['Tx. classificados como 1 corretamente'] = nos['Evento acumulado'] / nos['Total acumulado']
    
    # % Resp. 0 acumulado
    nos['Tx. classificados como 0 corretamente'] = (
     (nos['Nao-evento'].sum() - nos['Nao-evento acumulado'])/
     (nos['Total'].sum() - nos['Total acumulado'])
     )
    # Sens
    nos['Sens.'] = nos['Evento acumulado']/nos['Evento'].sum()
            
    # Ganho nos casos positivos
    nos["Ganho 1's"] = nos['Sens.'] - nos["% Total acumulado"]    
            
    # Espec
    nos['Espec.'] = (
        (nos['Nao-evento'].sum() - nos['Nao-evento acumulado'])/
        nos['Nao-evento'].sum()
        )

    # Ganho nos casos negtivos
    nos["Ganho 0's"] =  nos['Espec.'] - (1 - nos["% Total acumulado"])  


    # 1 - Espec
    nos['1 - Espec.'] = 1 - nos['Espec.']


    # 1 - Sens
    nos['1 - Sens.'] = 1 - nos['Sens.']

    # Sens + Espec - 1
    nos['Sens. + Espec. - 1'] = nos['Sens.'] + nos['Espec.'] - 1

    # Acurácia
    nos['Acurácia'] = (
        (nos["Evento acumulado"] + nos['Nao-evento'].sum() - nos['Nao-evento acumulado'])/
        nos["Total"].sum()
        )
    
    # KS
    nos['KS'] = nos['Sens. + Espec. - 1'].max()
    nos['KS2'] = ks_2samp(y_prob[y==1], y_prob[y!=1]).statistic
    
    alpha  = 0.05
    prob_0 = y_prob[y==0] 
    prob_1 = y_prob[y==1]
    ks, p_value = ks_test(prob_0, prob_1, alpha)
    nos['KS3'] = ks

    nos.reset_index(inplace=True)
    nos['index'] += 1
    nos.rename(columns={'index': 'Faixa'}, inplace=True)

    nos.rename(columns={'Nao-evento': '0',
                        'Evento': '1',
                        'tx_resposta': 'Prob',
                        'node': 'No'},
               inplace=True)
    
    nos.fillna(0, inplace = True)

    nos = nos[[
        "Faixa", 
        "Prob", 
        "Evento acumulado", 
        "Nao-evento acumulado", 
        #"1", 
        #"0",
        #"Total", 
        #"Total acumulado", 
        "% Total acumulado",
        "Tx. classificados como 1 corretamente",
        "Tx. classificados como 0 corretamente", 
        "Sens.", 
        "Ganho 1's", 
        "Espec.",
        "Ganho 0's", 
        #"1 - Espec.", 
        #"1 - Sens.", 
        "Sens. + Espec. - 1",
        "Acurácia", 
        #"KS", 
        #"KS2", 
        #"KS3"
        ]]

    return nos

def show_df(df, cols_to_percent = None):
    
    if cols_to_percent != None:
        df_styled = (
            df.style
                # Cor do header e index
                .set_table_styles([{
                    'selector': 'th:not(.index_name)',
                    'props': f'background-color: {COLOR}; color: white; text-align: center;'
                }]) 
                # Alinhamento, largura das colunas, cor da tabela, cor da letra
                .set_properties(**{'text-align': 'center'})
                .format('{:.0%}', subset= cols_to_percent) 
                .hide(axis="index")
            )
        
    else:
        df_styled = (
            df.style
                # Cor do header e index
                .set_table_styles([{
                    'selector': 'th:not(.index_name)',
                    'props': f'background-color: {COLOR}; color: white; text-align: center;'
                }]) 
                # Alinhamento, largura das colunas, cor da tabela, cor da letra
                .set_properties(**{'text-align': 'center'})
                .hide(axis="index")
            )
        
    return df_styled


def create_confusion_matrix(
        real,
        predict,
        event_class_real,       
        nonevent_class_real,    
        event_class_predict,    
        nonevent_class_predict, 
        new_name_real    = None,         
        new_name_predict = None):
    

    real             = np.array(real) 
    predict          = np.array(predict)
    confusion_matrix = pd.crosstab(real, predict)
    
    # Validacao
    c1 = event_class_predict    not in confusion_matrix.columns
    c2 = nonevent_class_predict not in confusion_matrix.columns
    
    c3 = event_class_real       not in confusion_matrix.index
    c4 = nonevent_class_real    not in confusion_matrix.index
    
    c5 = event_class_predict == nonevent_class_predict 
    c6 = event_class_real    == nonevent_class_real
    
    c7 = confusion_matrix.shape != (2,2)
    
    error = "\n"
    if c1: error += ">>> The value entered in event_class_predict is not in predict\n"
    if c2: error += ">>> The value entered in nonevent_class_predict is not in predict\n"
    if c3: error += ">>> The value entered in event_class_real is not in real\n"
    if c4: error += ">>> The value entered in nonevent_class_real is not in real\n"
    
    if c5: error += ">>> The event_class_predict and nonevent_class_predict are equals\n"
    if c6: error += ">>> The event_class_real and nonevent_class_real are equals\n"
    
    if c7: error += ">>> The generated confusion matrix is ​​not 2x2\n"
    
    if len(error) > 1:
        raise ValueError(error)
            
    # ------------------ Ordenando as colunas ---------------------------------
    confusion_matrix.reset_index(inplace = True)
    confusion_matrix.columns = [""] + list(confusion_matrix.columns)[1:]
    confusion_matrix = confusion_matrix[[
        "", 
        event_class_predict, 
        nonevent_class_predict
        ]]


    # ------------------ Ordenando as linhas ---------------------------------
    c = confusion_matrix[""] == event_class_real
    confusion_matrix["idx"]  = np.where(c, 1, 0)
    confusion_matrix.sort_values(by = "idx", ascending = False, inplace = True)
    confusion_matrix.drop("idx", axis = 1, inplace = True)

    # ---------------------------- Renomeando ---------------------------------
    if new_name_real != None:
        confusion_matrix[""] =  confusion_matrix[""].astype(str) + " " + new_name_real
        
    if new_name_predict != None:
        confusion_matrix.rename(columns = {
            event_class_predict: f"{event_class_predict} {new_name_predict}",
            nonevent_class_predict: f"{nonevent_class_predict} {new_name_predict}",
            }, inplace = True)
        
    # ---------------------------- Totais ---------------------------------
    total_by_col = ["Total"] + list(confusion_matrix.iloc[:, 1:].sum().values) 
    
    confusion_matrix = (
        pd
        .concat([
            confusion_matrix,  
            pd.DataFrame({
                confusion_matrix.columns[0] : total_by_col[0], 
                confusion_matrix.columns[1] : total_by_col[1],
                confusion_matrix.columns[2] : total_by_col[2]
            }, index = [0])], 
            axis = 0
            )
        )
    
    confusion_matrix["Total"] = confusion_matrix.iloc[:, 1:].sum(axis = 1)
    confusion_matrix.reset_index(drop = True, inplace = True)
    
    return confusion_matrix
    

def show_confusion_matrix(
        real,
        predict,
        event_class_real,       
        nonevent_class_real,    
        event_class_predict,    
        nonevent_class_predict, 
        new_name_real    = None,         
        new_name_predict = None):
    
    df = create_confusion_matrix(
            real,
            predict,
            event_class_real,       
            nonevent_class_real,    
            event_class_predict,    
            nonevent_class_predict,
            new_name_real,
            new_name_predict
            )
    
    px    = '500px'
    idx_col   = list(df.columns)
    
    df_styled = (
        df.style
            # Header and index color
            .set_table_styles([{
                'selector': 'th:not(.index_name)',
                'props': f'background-color: {COLOR}; color: white; text-align: center;'
            }]) 
            # Alignment, columns width, background color, text color
            .set_properties(subset = idx_col[0],
                            **{'text-align'      : 'center',
                               'background-color': COLOR,
                               'color'           : 'white',
                               'font-weight'     : 'bold',
                               'width'           : px,
                               })
            .set_properties(subset = idx_col[-1],
                            **{'text-align'      : 'center',
                               'background-color': "#C5D9F1",
                               'color'           : 'black',
                               'font-weight'     : 'bold',
                               'width'           : px,
                               })
            .set_properties(subset=([2], idx_col[1:]),
                            **{'text-align'      : 'center',
                               'background-color': "#C5D9F1",
                               'color'           : 'black',
                               'font-weight'     : 'bold',
                               'width'           : px,
                               })
            .set_properties(subset = ([0], idx_col[1]),
                            **{'text-align'      : 'center',
                               'width'           : px,
                               'background-color':"#89D329",
                               'color'           : "black",
                               })
            .set_properties(subset = ([0], idx_col[2]),
                            **{'text-align'      : 'center',
                               'width'           : px,
                               'background-color': "#FF3162",
                               'color'           : "black",
    
                               })
            .set_properties(subset = ([1], idx_col[1]),
                            **{'text-align'      : 'center',
                               'width'           : px,
                               'background-color':"#FF3162",
                               'color'           : "black",
                               })
            .set_properties(subset = ([1], idx_col[2]),
                            **{'text-align'      :'center',
                               'width'           : px,
                               'background-color': "#89D329",
                               'color'           : "black",
           
                               })
            .hide(axis="index")
        )
        
    return df_styled
