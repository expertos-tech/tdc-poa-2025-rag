import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'tdc-ai',
    pathMatch: 'full',
  },
  {
    path: 'tdc-ai',
    loadComponent: () =>
      import('./tdc-ai-assistant/tdc-ai-assistant.component').then(
        (m) => m.TdcAiAssistantComponent
      ),
  }
];
