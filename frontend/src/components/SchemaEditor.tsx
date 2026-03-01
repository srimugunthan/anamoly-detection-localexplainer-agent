"use client";

import { useApp } from "@/context/AppContext";
import type { CategoricalFeatureDef, FloatFeatureDef } from "@/types";

export default function SchemaEditor() {
  const { state } = useApp();
  if (!state.schema) return null;

  const features = Object.entries(state.schema.features);

  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
        Schema Preview — {features.length} features
      </h3>
      <div className="overflow-hidden rounded-md border border-gray-200">
        <table className="w-full text-xs">
          <thead className="bg-gray-50 text-left text-gray-500">
            <tr>
              <th className="px-3 py-2 font-medium">Feature</th>
              <th className="px-3 py-2 font-medium">Type</th>
              <th className="px-3 py-2 font-medium">Range / Values</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {features.map(([name, def]) => (
              <tr key={name} className="hover:bg-gray-50">
                <td className="px-3 py-2 font-mono font-medium text-gray-800">
                  {name}
                </td>
                <td className="px-3 py-2">
                  <span
                    className={`inline-block rounded px-1.5 py-0.5 font-medium ${
                      def.type === "categorical"
                        ? "bg-purple-100 text-purple-700"
                        : "bg-blue-100 text-blue-700"
                    }`}
                  >
                    {def.type}
                  </span>
                </td>
                <td className="px-3 py-2 text-gray-600">
                  {def.type === "categorical" ? (
                    (def as CategoricalFeatureDef).values.join(", ")
                  ) : (
                    <>
                      {(def as FloatFeatureDef).min} &ndash;{" "}
                      {(def as FloatFeatureDef).max}
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
