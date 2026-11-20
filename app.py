import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
from datetime import datetime, date

# --- CONFIGURACIÓN MOBILE-FIRST ---
st.set_page_config(
    page_title="Seguimiento Preventa DAS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CONEXIÓN SUPABASE ---
@st.cache_resource
def init_supabase() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = init_supabase()

# --- FUNCIONES DE NORMALIZACIÓN UNIFLEX ---
def normalizar_doc(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    # Convertir RUC 10 (11 dígitos) a DNI (8 dígitos) para cruce exacto
    if len(digits) == 11 and digits.startswith("10"):
        return digits[2:10]
    if len(digits) < 8:
        return digits.zfill(8)
    return digits

def clasificar_corte(hora_val):
    if pd.isna(hora_val):
        return "3. Corte Tarde (>15:00)"
    h_str = str(hora_val).strip()[:5]
    if h_str < "11:00":
        return "1. Corte Mañana (<11:00)"
    elif h_str <= "15:00":
        return "2. Corte Mediodía (11:00-15:00)"
    return "3. Corte Tarde (>15:00)"

def procesar_maestro(file) -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name=0)
    # Eliminar columna duplicada CodigoBees.1 si existe
    df = df.loc[:, ~df.columns.duplicated()]
    if "CodigoBees.1" in df.columns:
        df = df.drop(columns=["CodigoBees.1"])

    df["doc_norm"] = df["DocIdentidad"].apply(normalizar_doc)
    df = df[df["doc_norm"].notna()].copy()

    # Priorizar Activo='SI' y CodigoBees ascendente para resolver multi-local sin duplicar llave primaria
    df["prio_activo"] = np.where(df["Activo"].astype(str).str.upper() == "SI", 0, 1)
    df = df.sort_values(["doc_norm", "prio_activo", "CodigoBees"]).drop_duplicates(subset=["doc_norm"], keep="first")

    out = pd.DataFrame({
        "doc_norm": df["doc_norm"],
        "codigo_bees": df["CodigoBees"].fillna("").astype(str).str.strip(),
        "codigo_da": df["CodigoDA"].fillna("").astype(str).str.replace(".0", "", regex=False).str.zfill(6),
        "cliente": df["Cliente"].fillna("").astype(str).str.strip(),
        "direccion": df["Direccion"].fillna("").astype(str).str.strip(),
        "distrito": df["Distrito"].fillna("").astype(str).str.strip(),
        "localidad": df["Localidad"].fillna("").astype(str).str.strip(),
        "latitud": pd.to_numeric(df["Latitud"], errors="coerce").fillna(0.0),
        "longitud": pd.to_numeric(df["Longitud"], errors="coerce").fillna(0.0),
        "canal": df["Canal"].fillna("").astype(str).str.strip(),
        "subcanal": df["Subcanal"].fillna("").astype(str).str.strip(),
        "tipo_negocio": df["TipoNegocio"].fillna("").astype(str).str.strip(),
        "supervisor": df["Supervisor"].fillna("SIN ASIGNAR").astype(str).str.strip(),
        "vendedor": df["Vendedor"].fillna("SIN ASIGNAR").astype(str).str.strip(),
        "codigo_zona": df["CodigoZona"].fillna("").astype(str).str.strip(),
        "zona": df["Zona"].fillna("").astype(str).str.strip(),
        "dia_visita": df["DiaVisita"].fillna("").astype(str).str.lower().str.strip(),
        "activo": df["Activo"].fillna("NO").astype(str).str.upper().str.strip(),
        "telefono": df["Telefono"].fillna("").astype(str).str.replace(".0", "", regex=False).str.strip(),
        "categoria_cliente": df["CategoriaCliente"].fillna("").astype(str).str.strip(),
        "linea_credito": pd.to_numeric(df["LineaCredito"], errors="coerce").fillna(0.0),
        "cliente_saldo": pd.to_numeric(df["ClienteSaldo"], errors="coerce").fillna(0.0),
    })
    return out

def procesar_ventas(file) -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name=0)
    df["doc_norm"] = df["Doc Identidad"].apply(normalizar_doc)

    f_emision = pd.to_datetime(df["FechaEmision"], errors="coerce").dt.strftime("%Y-%m-%d")
    f_venta = pd.to_datetime(df["FechaVenta"], errors="coerce").dt.strftime("%Y-%m-%d")
    hora_str = df["Hora"].fillna("").astype(str).str.strip()

    out = pd.DataFrame({
        "periodo": pd.to_numeric(df["Periodo"], errors="coerce").fillna(0).astype(int),
        "fecha_emision": f_emision.where(f_emision.notna(), None),
        "fecha_venta": f_venta.where(f_venta.notna(), None),
        "hora": hora_str,
        "franja_corte": hora_str.apply(clasificar_corte),
        "tipo_doc": df["TipoDoc"].fillna("").astype(str).str.strip(),
        "numero_pedido": df["Numero"].fillna("").astype(str).str.strip(),
        "cp_venta": df["CpVenta"].fillna("").astype(str).str.strip(),
        "estado": df["Estado"].fillna("").astype(str).str.upper().str.strip(),
        "estado_aprobacion": df["EstadoAprobacion"].fillna("").astype(str).str.upper().str.strip(),
        "estado_entrega": df["EstadoEntrega"].fillna("").astype(str).str.upper().str.strip(),
        "estado_facturacion": df["EstadoFacturacion"].fillna("").astype(str).str.upper().str.strip(),
        "doc_norm": df["doc_norm"],
        "codigo_cliente": df["CodigoCliente"].fillna("").astype(str).str.replace(".0", "", regex=False).str.zfill(6),
        "nombre_cliente_venta": df["NombreCliente"].fillna("").astype(str).str.strip(),
        "distrito_venta": df["Distrito"].fillna("").astype(str).str.strip(),
        "condicion_pago": df["CondicionPago"].fillna("").astype(str).str.strip(),
        "cod_agente": df["Cod Agente"].fillna("").astype(str).str.replace(".0", "", regex=False).str.zfill(5),
        "agente_comercial": df["AgenteComercial"].fillna("").astype(str).str.strip(),
        "zona_venta": df["Zona Venta"].fillna("").astype(str).str.strip(),
        "codigo_producto": df["CodigoProducto"].fillna("").astype(str).str.replace(".0", "", regex=False).str.strip(),
        "descripcion_producto": df["DescripcionProducto"].fillna("").astype(str).str.strip(),
        "subcategoria": df["SubCategoria"].fillna("").astype(str).str.strip(),
        "linea": df["Linea"].fillna("").astype(str).str.strip(),
        "marca": df["Marca"].fillna("").astype(str).str.strip(),
        "bonificacion": df["Bonificacion"].fillna("NO").astype(str).str.upper().str.strip(),
        "cantidad": pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0.0),
        "item_monto": pd.to_numeric(df["Item Monto"], errors="coerce").fillna(0.0),
        "hl": pd.to_numeric(df["HL"], errors="coerce").fillna(0.0),
        "cajas_convertidas": pd.to_numeric(df["CajasConvertidas"], errors="coerce").fillna(0.0),
    })
    return out

def subir_por_lotes(tabla: str, registros: list, modo: str = "upsert", conflicto: str = "doc_norm"):
    chunk_size = 1000
    total = len(registros)
    barra = st.progress(0)
    for i in range(0, total, chunk_size):
        lote = registros[i : i + chunk_size]
        if modo == "upsert":
            supabase.table(tabla).upsert(lote, on_conflict=conflicto).execute()
        else:
            supabase.table(tabla).insert(lote).execute()
        barra.progress(min((i + chunk_size) / total, 1.0))

# --- CONSULTAS CON CACHÉ (OPTIMIZADO PARA BAJA SEÑAL) ---
@st.cache_data(ttl=300)
def cargar_periodos():
    res = supabase.table("fact_ventas").select("periodo").limit(1000).execute()
    periodos = sorted(list({r["periodo"] for r in res.data}), reverse=True)
    return periodos if periodos else [int(datetime.now().strftime("%Y%m"))]

@st.cache_data(ttl=300)
def cargar_ventas_periodo(periodo: int) -> pd.DataFrame:
    rows = []
    offset = 0
    step = 1000
    while True:
        res = (
            supabase.table("vw_seguimiento_preventa")
            .select("*")
            .eq("periodo", periodo)
            .range(offset, offset + step - 1)
            .execute()
        )
        if not res.data:
            break
        rows.extend(res.data)
        if len(res.data) < step:
            break
        offset += step
    return pd.DataFrame(rows)

@st.cache_data(ttl=600)
def cargar_maestro_activo() -> pd.DataFrame:
    rows = []
    offset = 0
    step = 1000
    while True:
        res = (
            supabase.table("dim_clientes")
            .select("doc_norm,codigo_bees,cliente,distrito,canal,supervisor,vendedor,zona,dia_visita,activo,telefono")
            .eq("activo", "SI")
            .range(offset, offset + step - 1)
            .execute()
        )
        if not res.data:
            break
        rows.extend(res.data)
        if len(res.data) < step:
            break
        offset += step
    return pd.DataFrame(rows)

# --- INTERFAZ PRINCIPAL ---
st.title("📊 Control de Preventa DAS")

tab1, tab2, tab3 = st.tabs(["⚡ Cortes Preventa", "🎯 Cobertura & Ruta", "⚙️ Actualizar BD"])

# ==========================================
# PESTAÑA 1: CORTES DE PREVENTA (3x DÍA)
# ==========================================
with tab1:
    periodos = cargar_periodos()
    c_f1, c_f2, c_f3 = st.columns([1, 2, 2])
    with c_f1:
        periodo_sel = st.selectbox("Periodo", periodos)
    df_v = cargar_ventas_periodo(periodo_sel)

    if df_v.empty:
        st.info("Aún no hay datos cargados para este periodo. Ve a la pestaña 'Actualizar BD'.")
    else:
        with c_f2:
            sups = ["TODOS"] + sorted(df_v["supervisor"].dropna().unique().tolist())
            sup_sel = st.selectbox("Supervisor", sups)
        df_sub = df_v if sup_sel == "TODOS" else df_v[df_v["supervisor"] == sup_sel]

        with c_f3:
            vends = ["TODOS"] + sorted(df_sub["vendedor_preventa"].dropna().unique().tolist())
            vend_sel = st.selectbox("Vendedor Preventa", vends)
        if vend_sel != "TODOS":
            df_sub = df_sub[df_sub["vendedor_preventa"] == vend_sel]

        # Filtro de fecha (Acumulado mes vs Día específico)
        fechas_disp = sorted(df_sub["fecha_emision"].dropna().unique().tolist(), reverse=True)
        modo_fecha = st.radio("Corte de consulta:", ["Acumulado Mes", "Solo Último Día / Fecha"], horizontal=True)
        if modo_fecha == "Solo Último Día / Fecha" and fechas_disp:
            fecha_sel = st.selectbox("Fecha Emisión", fechas_disp)
            df_sub = df_sub[df_sub["fecha_emision"] == fecha_sel]

        # Separar EMITIDO (Preventa real) vs ANULADO
        emitido = df_sub[df_sub["estado"] == "EMITIDO"]
        facturado = emitido[emitido["estado_entrega"] == "TOTAL"]
        pendiente = emitido[emitido["estado_entrega"] != "TOTAL"]
        anulado = df_sub[df_sub["estado"] == "ANULADO"]

        # KPIs Principales
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("HL Preventa (Emitido)", f"{emitido['hl'].sum():,.2f} HL", f"Pend: {pendiente['hl'].sum():,.2f} HL")
        k2.metric("Cajas Convertidas", f"{emitido['cajas_convertidas'].sum():,.0f}")
        k3.metric("Venta Neta (S/)", f"S/ {emitido['item_monto'].sum():,.0f}")
        clientes_unicos = emitido[emitido["doc_norm"] != "00000001"]["doc_norm"].nunique()
        pedidos_unicos = emitido["numero_pedido"].nunique()
        k4.metric("Clientes / Pedidos", f"{clientes_unicos:,} cli", f"{pedidos_unicos:,} pedidos")

        st.divider()

        # Avance por Franja Horaria (3 cortes)
        st.subheader("🕒 Avance por Franja de Corte")
        cortes_df = (
            emitido.groupby("franja_corte", as_index=False)
            .agg(
                Pedidos=("numero_pedido", "nunique"),
                Clientes=("doc_norm", "nunique"),
                HL=("hl", "sum"),
                Cajas=("cajas_convertidas", "sum"),
                Soles=("item_monto", "sum"),
            )
            .sort_values("franja_corte")
        )
        st.dataframe(
            cortes_df.style.format({"HL": "{:,.2f}", "Cajas": "{:,.1f}", "Soles": "S/ {:,.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

        # Resumen por Línea (Cerveza vs NABS) y Estado
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🍺 Mix Cerveza vs. NABS")
            mix_df = (
                emitido.groupby("linea", as_index=False)
                .agg(HL=("hl", "sum"), Cajas=("cajas_convertidas", "sum"), Soles=("item_monto", "sum"))
            )
            st.dataframe(
                mix_df.style.format({"HL": "{:,.2f}", "Cajas": "{:,.1f}", "Soles": "S/ {:,.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

        with col_b:
            st.subheader("📦 Estatus de Pedidos")
            est_df = (
                df_sub.groupby(["estado", "estado_entrega"], as_index=False)
                .agg(Pedidos=("numero_pedido", "nunique"), HL=("hl", "sum"), Soles=("item_monto", "sum"))
            )
            st.dataframe(
                est_df.style.format({"HL": "{:,.2f}", "Soles": "S/ {:,.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

        # Ranking por Vendedor
        st.subheader("🏆 Avance por Vendedor")
        rank_df = (
            emitido.groupby(["supervisor", "vendedor_preventa"], as_index=False)
            .agg(
                Clientes=("doc_norm", "nunique"),
                Pedidos=("numero_pedido", "nunique"),
                HL=("hl", "sum"),
                Cajas=("cajas_convertidas", "sum"),
                Soles=("item_monto", "sum"),
            )
            .sort_values("HL", ascending=False)
        )
        st.dataframe(
            rank_df.style.format({"HL": "{:,.2f}", "Cajas": "{:,.1f}", "Soles": "S/ {:,.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

# ==========================================
# PESTAÑA 2: COBERTURA Y PENDIENTES DE RUTA
# ==========================================
with tab2:
    df_m = cargar_maestro_activo()
    periodos = cargar_periodos()
    df_v_mes = cargar_ventas_periodo(periodos[0]) if periodos else pd.DataFrame()

    if df_m.empty or df_v_mes.empty:
        st.warning("Carga primero el Maestro y las Ventas en la pestaña 'Actualizar BD'.")
    else:
        # Excluir CLIENTES VARIOS (00000001)
        base_activa = df_m[df_m["doc_norm"] != "00000001"].copy()
        compradores_mes = set(
            df_v_mes[(df_v_mes["estado"] == "EMITIDO") & (df_v_mes["doc_norm"] != "00000001")]["doc_norm"].dropna()
        )
        base_activa["compro_mes"] = base_activa["doc_norm"].isin(compradores_mes)

        c1, c2, c3 = st.columns(3)
        sup_m = c1.selectbox("Supervisor Cartera", ["TODOS"] + sorted(base_activa["supervisor"].unique().tolist()))
        df_cob = base_activa if sup_m == "TODOS" else base_activa[base_activa["supervisor"] == sup_m]

        vend_m = c2.selectbox("Vendedor Cartera", ["TODOS"] + sorted(df_cob["vendedor"].unique().tolist()))
        if vend_m != "TODOS":
            df_cob = df_cob[df_cob["vendedor"] == vend_m]

        dias_sem = ["TODOS", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"]
        dia_sel = c3.selectbox("Filtrar por Día de Visita", dias_sem)
        if dia_sel != "TODOS":
            df_cob = df_cob[df_cob["dia_visita"].str.contains(dia_sel, na=False)]

        tot_activos = len(df_cob)
        tot_comp = int(df_cob["compro_mes"].sum())
        pct_cob = (tot_comp / tot_activos * 100) if tot_activos > 0 else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Cartera Activa", f"{tot_activos:,}")
        m2.metric("Compradores del Mes", f"{tot_comp:,}")
        m3.metric("% Cobertura", f"{pct_cob:.1f}%")

        st.subheader("📋 Clientes Activos SIN Compra en el Mes (Para gestión en ruta)")
        sin_compra = df_cob[~df_cob["compro_mes"]][
            ["codigo_bees", "doc_norm", "cliente", "distrito", "zona", "dia_visita", "telefono"]
        ]
        st.dataframe(sin_compra, use_container_width=True, hide_index=True)

        csv_data = sin_compra.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Descargar Lista Pendientes (CSV ligero)", csv_data, "pendientes_cobertura.csv", "text/csv")

# ==========================================
# PESTAÑA 3: CARGA DE ARCHIVOS UNIFLEX (ADMIN)
# ==========================================
with tab3:
    st.subheader("🔄 Actualización de Bases desde Uniflex")
    pwd = st.text_input("Clave de Administrador", type="password")

    if pwd == st.secrets["ADMIN_PASS"]:
        col_up1, col_up2 = st.columns(2)

        with col_up1:
            st.markdown("### 1. Maestro de Clientes (Semanal)")
            f_maestro = st.file_uploader("Sube MAESTRO_DAS.xlsx", type=["xlsx"], key="up_m")
            if f_maestro and st.button("Procesar y Subir Maestro"):
                with st.spinner("Limpiando Maestro y deduplicando CodigoBees..."):
                    df_m_clean = procesar_maestro(f_maestro)
                    subir_por_lotes("dim_clientes", df_m_clean.to_dict(orient="records"), modo="upsert", conflicto="doc_norm")
                    st.cache_data.clear()
                st.success(f"✅ Maestro actualizado: {len(df_m_clean):,} clientes únicos cargados.")

        with col_up2:
            st.markdown("### 2. Ventas / Preventa (3 veces al día)")
            f_ventas = st.file_uploader("Sube VENTAS_DA.xlsx", type=["xlsx"], key="up_v")
            if f_ventas and st.button("Procesar y Subir Corte de Ventas"):
                with st.spinner("Normalizando RUC/DNI y reemplazando periodo activo..."):
                    df_v_clean = procesar_ventas(f_ventas)
                    periodos_archivo = df_v_clean["periodo"].unique().tolist()
                    for p in periodos_archivo:
                        supabase.table("fact_ventas").delete().eq("periodo", int(p)).execute()
                    subir_por_lotes("fact_ventas", df_v_clean.to_dict(orient="records"), modo="insert")
                    st.cache_data.clear()
                st.success(f"✅ Corte actualizado: {len(df_v_clean):,} líneas cargadas para periodo(s) {periodos_archivo}.")
    elif pwd:
        st.error("Clave incorrecta.")
