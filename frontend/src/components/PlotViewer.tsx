"use client";

import { useState } from "react";
import Image from "next/image";
import { useApp } from "@/context/AppContext";

type Tab = "shap" | "lime" | "pdp";

export default function PlotViewer() {
  const { state } = useApp();
  const [activeTab, setActiveTab] = useState<Tab>("shap");
  const [lightbox, setLightbox] = useState<string | null>(null);

  if (!state.explainResult) return null;

  const { plots } = state.explainResult.explanations;
  const hasShap = plots.shap_plot_url || plots.shap_force_plot_url;
  const hasLime = !!plots.lime_plot_url;
  const hasPdp = plots.pdp_plot_urls.length > 0;

  const tabs: { key: Tab; label: string; available: boolean }[] = [
    { key: "shap", label: "SHAP", available: !!hasShap },
    { key: "lime", label: "LIME", available: hasLime },
    { key: "pdp", label: "PDP", available: hasPdp },
  ];

  function PlotImg({ url, alt }: { url: string; alt: string }) {
    return (
      <div
        className="relative cursor-zoom-in overflow-hidden rounded-md border border-gray-200 bg-white"
        onClick={() => setLightbox(url)}
      >
        <Image
          src={url}
          alt={alt}
          width={700}
          height={400}
          className="w-full object-contain"
          unoptimized
        />
      </div>
    );
  }

  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-gray-500">
        Explanation Plots
      </h3>

      {/* Tabs */}
      <div className="mb-4 flex gap-1 rounded-lg bg-gray-100 p-1">
        {tabs.map((t) => (
          <button
            key={t.key}
            type="button"
            disabled={!t.available}
            onClick={() => setActiveTab(t.key)}
            className={`flex-1 rounded-md py-1.5 text-sm font-medium transition-colors ${
              activeTab === t.key
                ? "bg-white text-indigo-700 shadow-sm"
                : t.available
                ? "text-gray-600 hover:text-gray-800"
                : "cursor-not-allowed text-gray-300"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="space-y-3">
        {activeTab === "shap" && (
          <>
            {plots.shap_plot_url && (
              <PlotImg url={plots.shap_plot_url} alt="SHAP waterfall" />
            )}
            {plots.shap_force_plot_url && (
              <PlotImg url={plots.shap_force_plot_url} alt="SHAP bar chart" />
            )}
            {!hasShap && (
              <p className="text-sm text-gray-400">No SHAP plots available.</p>
            )}
          </>
        )}

        {activeTab === "lime" && (
          <>
            {plots.lime_plot_url ? (
              <PlotImg url={plots.lime_plot_url} alt="LIME bar chart" />
            ) : (
              <p className="text-sm text-gray-400">No LIME plot available.</p>
            )}
          </>
        )}

        {activeTab === "pdp" && (
          <>
            {plots.pdp_plot_urls.length > 0 ? (
              plots.pdp_plot_urls.map((url, i) => (
                <PlotImg key={url} url={url} alt={`PDP feature ${i + 1}`} />
              ))
            ) : (
              <p className="text-sm text-gray-400">No PDP plots available.</p>
            )}
          </>
        )}
      </div>

      {/* Lightbox */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4"
          onClick={() => setLightbox(null)}
        >
          <div className="relative max-h-full max-w-5xl overflow-auto rounded-lg bg-white p-2">
            <Image
              src={lightbox}
              alt="Full size plot"
              width={1200}
              height={800}
              className="max-h-[90vh] w-auto object-contain"
              unoptimized
            />
            <button
              type="button"
              className="absolute right-3 top-3 rounded-full bg-gray-800/70 px-2.5 py-0.5 text-xs text-white hover:bg-gray-800"
              onClick={() => setLightbox(null)}
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
