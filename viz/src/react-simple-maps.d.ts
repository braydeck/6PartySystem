declare module 'react-simple-maps' {
  import { ComponentType, ReactNode } from 'react';

  export interface GeographiesChildrenProps {
    geographies: Geography[];
  }

  export interface Geography {
    rsmKey: string;
    id?: string | number;
    [key: string]: unknown;
  }

  export interface ComposableMapProps {
    projection?: string;
    style?: React.CSSProperties;
    children?: ReactNode;
    [key: string]: unknown;
  }

  export interface GeographiesProps {
    geography: string | object;
    children: (props: GeographiesChildrenProps) => ReactNode;
    [key: string]: unknown;
  }

  export interface GeographyProps {
    geography: Geography;
    fill?: string;
    stroke?: string;
    strokeWidth?: number;
    style?: {
      default?: React.CSSProperties;
      hover?: React.CSSProperties;
      pressed?: React.CSSProperties;
    };
    onMouseEnter?: () => void;
    onMouseLeave?: () => void;
    [key: string]: unknown;
  }

  export const ComposableMap: ComponentType<ComposableMapProps>;
  export const Geographies: ComponentType<GeographiesProps>;
  export const Geography: ComponentType<GeographyProps>;
  export const ZoomableGroup: ComponentType<{ children?: ReactNode; [key: string]: unknown }>;
}
