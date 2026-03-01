"use client";

import { AppProvider } from "@/context/AppContext";
import ModelUpload from "@/components/ModelUpload";
import SchemaEditor from "@/components/SchemaEditor";
import RecordInput from "@/components/RecordInput";
import PredictionBadge from "@/components/PredictionBadge";
import PlotViewer from "@/components/PlotViewer";
import FeatureTable from "@/components/FeatureTable";
import ExplanationSummary from "@/components/ExplanationSummary";
import LoadingState from "@/components/LoadingState";

export default function Home() {
  return (
    <AppProvider>
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="border-b border-gray-200 bg-white px-6 py-4 shadow-sm">
          <div className="mx-auto flex max-w-7xl items-center justify-between">
            <div>
              <h1 className="text-xl font-bold text-gray-900">
                Anomaly Explainer
              </h1>
              <p className="text-xs text-gray-500">
                SHAP · LIME · PDP · LLM interpretation
              </p>
            </div>
            <span className="rounded-full bg-indigo-100 px-3 py-1 text-xs font-medium text-indigo-700">
              Local Explanation Agent
            </span>
          </div>
        </header>

        {/* Main layout */}
        <main className="mx-auto max-w-7xl px-6 py-8">
          <div className="flex gap-8">
            {/* LEFT: Upload + input panel */}
            <aside className="w-80 flex-shrink-0 space-y-6">
              <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
                <ModelUpload />
              </div>

              <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
                <SchemaEditor />
              </div>

              <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
                <RecordInput />
              </div>
            </aside>

            {/* RIGHT: Results panel */}
            <section className="min-w-0 flex-1 space-y-6">
              <LoadingState />

              <PredictionBadge />
              <ExplanationSummary />
              <PlotViewer />
              <FeatureTable />
            </section>
          </div>
        </main>
      </div>
    </AppProvider>
  );
}
